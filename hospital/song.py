"""'Sinus Rhythm': an EDM track made only from synthesised hospital machines, sung in Telemetric.

  python -m hospital.song                 # -> hospital/out/sinus_rhythm.wav (+ .mp3 if ffmpeg exists)
  python -m hospital.song --samples       # also writes the machine sample pack to hospital/out/samples/

128 BPM, 84 bars. Key: C = 240 Hz (4 x 60 Hz mains). Progression Am - F - C - G.
At 128 BPM a 16th note is 117 ms, which is what makes the medical timings fit the grid:
the lup-dup gap, an IEC high-priority burst (exactly one bar), and a two-bar ventilator breath
(16 breaths/min).
"""
import argparse
import os
import shutil
import subprocess
import wave

import numpy as np

from . import language as L
from . import machines as M
from .dsp import SR, convolve_stereo, hp, lp, n_, noise, pan, reverb_ir, tv_filter

BPM = 128
BEAT = 60 / BPM
BAR = 4 * BEAT
S16 = BEAT / 4
N_BARS = 84
TAIL = 4.0

# (bass root semitones from c, pad voicing) for Am F C G
CHORDS = [(-27, [-3, 0, 4, 7]), (-31, [-7, -3, 0, 4]), (-24, [-5, 0, 4, 7]), (-29, [-5, -1, 2, 7])]
# the machines' conversation in drop 2: each word sits on the chord it spells
CONVERSATION = ["cardiac", "ventilation", "drug", "oxygen"]

SECTIONS = {"intro": (0, 8), "build": (8, 16), "drop1": (16, 32), "breakdown": (32, 44),
            "flatline": (44, 48), "drop2": (48, 72), "outro": (72, 84)}


def at(bar, beat=0.0, s16=0.0):
    return bar * BAR + beat * BEAT + s16 * S16


def in_(name, bar):
    a, b = SECTIONS[name]
    return a <= bar < b


class Mix:
    DUCK = {"drums": 0.0, "bass": 0.85, "pad": 0.55, "voice": 0.3, "fx": 0.25, "perc": 0.15, "kick": 0.0}

    def __init__(self, sec):
        self.n = n_(sec)
        self.stems = {k: np.zeros((2, self.n)) for k in self.DUCK}
        self.send = np.zeros((2, self.n))
        self.kicks = []

    def add(self, x, t, stem="fx", gain=1.0, p=0.0, send=0.0):
        i = n_(t)
        if i >= self.n or len(x) == 0:
            return
        x = x[: self.n - i] * gain
        st = pan(x, p)
        self.stems[stem][:, i:i + st.shape[1]] += st
        if send:
            self.send[:, i:i + st.shape[1]] += st * send

    def sidechain(self):
        imp = np.zeros(self.n)
        for t in self.kicks:
            i = n_(t)
            if i < self.n:
                imp[i] = 1.0
        tk = np.arange(n_(0.4)) / SR
        kern = np.minimum(tk / 0.004, 1) * np.exp(-tk / 0.13)
        sc = np.minimum(1.0, np.convolve(imp, kern)[: self.n])
        return sc

    def master(self, drive=1.4, wet_gain=0.55, ceiling_db=-1.0):
        sc = self.sidechain()
        out = np.zeros((2, self.n))
        for k, s in self.stems.items():
            out += s * (1 - self.DUCK[k] * sc)
        wet = convolve_stereo(self.send, reverb_ir(3.2))
        wet *= (1 - 0.4 * sc)
        out += wet * wet_gain
        out = np.vstack([hp(c, 28, order=2) for c in out])
        # gentle glue: normalise, soft-clip, normalise to -1 dBFS
        out /= np.percentile(np.abs(out), 99.9) + 1e-9
        out = np.tanh(drive * out) / np.tanh(drive)
        out *= 10 ** (ceiling_db / 20) / (np.abs(out).max() + 1e-9)
        return out


def build(mix):
    rng = np.random.default_rng(128)

    # ------------------------------------------------------------ mains hum (bass) and pads
    for bar in range(N_BARS):
        root, voicing = CHORDS[bar % 4]
        t = at(bar)
        if in_("intro", bar) or in_("outro", bar) and bar >= 78:
            fade = (bar + 1) / 8 if bar < 8 else max(0.0, (N_BARS - bar) / 6)
            mix.add(M.mains_hum(60.0, BAR, buzz=0.2), t, "bass", 0.16 * fade)
        elif in_("build", bar):
            if bar >= 12:
                mix.add(lp(M.mains_hum(M.nt(root), BAR), 200 + 120 * (bar - 12)), t, "bass", 0.3)
        elif in_("drop1", bar) or in_("drop2", bar) or (in_("outro", bar) and bar < 78):
            mix.add(M.mains_hum(M.nt(root), BAR), t, "bass", 0.36)
        elif in_("breakdown", bar) and bar >= 36:
            mix.add(M.mains_hum(M.nt(root), BAR, buzz=0.05), t, "bass", 0.28)

        pad_on = bar >= 4 and not in_("flatline", bar)
        if pad_on:
            g = 0.16 if bar < 8 else 0.2 if in_("build", bar) else 0.26
            if in_("breakdown", bar):
                g = 0.34
            for semi in voicing:
                x = M.alarm_pad_tone(M.nt(semi), BAR, attack=0.3 if not in_("breakdown", bar) else 0.6)
                # the pad breathes with the ventilator (2-bar cycle: 1/3 inhale swell, 2/3 exhale)
                tb = (np.arange(len(x)) / SR + (bar % 2) * BAR) % (2 * BAR)
                breath = np.where(tb < 1.0, tb / 1.0, np.exp(-(tb - 1.0) / 2.0))
                x *= 0.55 + 0.45 * breath
                if in_("intro", bar):
                    x = lp(x, 500)
                elif in_("build", bar):
                    x = lp(x, 700 + 450 * (bar - 8))
                elif in_("outro", bar):
                    x = lp(x, max(500, 4000 - 350 * (bar - 72)))
                mix.add(x, t, "pad", g, p=rng.uniform(-0.5, 0.5), send=0.35)

    # ------------------------------------------------------------ ward ambience
    for bar in range(0, N_BARS, 2):
        quiet = in_("drop1", bar) or in_("drop2", bar) or in_("flatline", bar)
        if not quiet:
            mix.add(M.ventilator_breath(), at(bar), "fx", 0.38, p=-0.15, send=0.25)
    mix.add(M.fetal_doppler(at(8) - at(2)), at(2), "fx", 0.06, p=-0.6, send=0.2)
    mix.add(M.fetal_doppler(at(4)), at(33), "fx", 0.05, p=0.6, send=0.2)
    for t in (at(0), at(14), at(46, 0), at(80)):
        mix.add(M.nurse_chime(), t, "fx", 0.18, p=0.35, send=0.6)
    for bar in list(range(4, 16)) + list(range(32, 44)) + list(range(56, 72)):
        for _ in range(3 if bar < 56 else 2):
            t = at(bar) + rng.uniform(0, BAR)
            mix.add(M.bubble(rng.uniform(380, 950)), t, "perc", 0.12, p=rng.uniform(-0.8, 0.8), send=0.3)

    # ------------------------------------------------------------ heart: QRS beeps and pulse-ox
    for bar in range(N_BARS):
        if in_("intro", bar) or (in_("outro", bar) and bar >= 76):
            beats = [0, 2]  # 64 BPM resting heart rate
        elif in_("build", bar):
            beats = [0, 1, 2, 3]  # 128 BPM
        elif in_("drop1", bar) or in_("drop2", bar):
            beats = [0, 1, 2, 3]
        elif in_("breakdown", bar) and bar >= 40:
            beats = [i / 2 for i in range(8)] if bar < 43 else [i / 4 for i in range(16)]
        else:
            beats = []
        g = 0.09 if (in_("drop1", bar) or in_("drop2", bar)) else 0.16
        for b in beats:
            mix.add(M.qrs_beep(), at(bar, b), "perc", g, p=0.2, send=0.15)

    # breakdown: desaturation. One beep per beat, SpO2 99 -> 76: the pitch falls an octave in quarter-tones.
    for k in range(32):
        spo2 = int(round(99 - 23 * k / 31))
        mix.add(M.pulseox_beep(spo2, 0.11), at(32 + k // 4, k % 4), "perc", 0.2, p=0.2, send=0.35)

    # ------------------------------------------------------------ drums
    def drum_bar(bar, full=True):
        for b in range(4):
            t = at(bar, b)
            mix.add(M.kick(), t, "kick", 1.0)
            mix.kicks.append(t)
            if b in (0, 2):  # the heartbeat's 'dup', one 16th after the 'lub'
                mix.add(M.heartbeat()[n_(0.117):], t + S16, "drums", 0.35)
        for b in (1, 3):
            mix.add(M.zap(), at(bar, b), "drums", 1.0, send=0.12)
        for s in range(16):
            vel = 0.5 if s % 4 == 2 else 0.22 if s % 2 else 0.3
            mix.add(M.pump_click(), at(bar, 0, s) + (0.012 if s % 2 else 0), "perc", vel * 0.6,
                    p=0.35 if s % 2 else -0.35)
        for b in range(4):
            mix.add(M.o2_hiss(0.13), at(bar, b + 0.5), "perc", 0.12, p=0.1)
        if full:
            for s, res in zip([3, 6, 11, 14], [720, 960, 720, 1080]):
                mix.add(M.mri_knock(res), at(bar, 0, s), "perc", 0.3, p=-0.4 if s < 8 else 0.4, send=0.1)

    for bar in range(N_BARS):
        if in_("drop1", bar) or in_("drop2", bar) or (in_("outro", bar) and bar < 76):
            drum_bar(bar, full=not (in_("outro", bar) or bar in (16, 17)))
        elif in_("build", bar):
            # half-time heartbeat, then every beat
            for b in ([0, 2] if bar < 12 else [0, 1, 2, 3]):
                mix.add(lp(M.heartbeat(), 900), at(bar, b), "kick", 0.85)
                mix.kicks.append(at(bar, b))
            step = 2 if bar < 12 else 1
            for s in range(0, 16, step):
                mix.add(M.pump_click(), at(bar, 0, s), "perc", 0.18 + 0.02 * (bar - 8))
        elif in_("intro", bar) and bar >= 4:
            for b in (0, 2):
                mix.add(lp(M.heartbeat(gap=0.2), 500), at(bar, b), "kick", 0.45, send=0.1)

    # MRI knock rolls: the scanner speeding up into each drop (resonance pitched up as a riser)
    def roll(bar0, bars, top=1.0):
        steps = []
        for k in range(bars * 16):
            bar, s = bar0 + k // 16, k % 16
            sub = 2 if (k >= bars * 16 - 8) else 1
            if k < (bars - 1) * 16 and s % 2:
                continue
            for j in range(sub):
                steps.append(at(bar, 0, s + j / sub))
        for i, t in enumerate(steps):
            p = i / len(steps)
            mix.add(M.mri_knock(720 * 2 ** p), t, "perc", (0.15 + 0.35 * p) * top, p=np.sin(i) * 0.3, send=0.15)
    roll(14, 2)
    roll(42, 2)
    roll(70, 2, 0.8)
    # fills
    for bar in (23, 31, 59, 63):
        for s in range(8, 16):
            mix.add(M.mri_knock(720 * 2 ** ((s - 8) / 8)), at(bar, 0, s), "perc", 0.25, p=0.3)

    # risers: oxygen hiss opening up, kettle coming to the boil
    for start, bars in ((12, 4), (68, 4)):
        sec = bars * BAR
        x = noise(sec)
        cut = 600 * (14000 / 600) ** (np.arange(len(x)) / len(x))
        x = tv_filter(x, cut, q=1.2, kind="hp") * np.linspace(0, 1, len(x)) ** 2
        mix.add(x, at(start), "fx", 0.16, send=0.3)
    mix.add(M.kettle(4 * BAR), at(40), "fx", 0.22, p=-0.25, send=0.4)

    # crashes: a long O2 blast + the defib shock on each drop downbeat
    for bar in (16, 48):
        mix.add(M.o2_hiss(3.0, 3000) * np.exp(-np.arange(n_(3.0)) / SR / 0.8), at(bar), "fx", 0.25, send=0.5)
    mix.add(M.defib_shock(), at(16), "drums", 0.5, send=0.3)

    # ------------------------------------------------------------ flatline -> charge -> SHOCK
    mix.add(M.flatline(BAR * 1.0), at(44), "fx", 0.22, send=0.5)
    mix.add(M.defib_charge(at(47, 2) - at(45)), at(45), "fx", 0.25, send=0.3)
    mix.add(M.flatline(BEAT * 1.4, f=1920.0) * 0.6, at(47, 2), "fx", 0.18)  # 'charged' tone
    mix.add(M.defib_shock(), at(48), "drums", 0.9, send=0.4)

    # EPI scanner stabs in the second half of each drop
    for bar in list(range(24, 32)) + list(range(56, 72)):
        for s in (7, 15) if bar < 64 else (2, 7, 10, 15):
            mix.add(lp(M.epi_buzz(0.06), 6000), at(bar, 0, s), "perc", 0.05, p=0.5 if s % 2 else -0.5, send=0.15)

    # dry alarm pulses doubling the talking lead, an octave up
    def dry(category, bar, gain):
        for on, semi in M.iec_melody(category, "high"):
            mix.add(M.iec_pulse(M.nt(semi + 12), 0.1), at(bar, 0, on), "voice", gain, p=0.3, send=0.25)

    for bar in range(24, 32):
        dry("cardiac", bar, 0.05)
    for bar in range(56, 72):
        dry(CONVERSATION[bar % 4], bar, 0.05 if bar < 64 else 0.07)
    # build call: the plain high-priority cardiac alarm, as a monitor would sound it
    for on, semi in M.iec_melody("cardiac", "high"):
        mix.add(M.iec_pulse(M.nt(semi), 0.11), at(15, 0, on), "fx", 0.2, send=0.4)

    # ------------------------------------------------------------ the voice (Telemetric)
    def say(sylls, gain, p=0.0, send=0.3, **kw):
        x, t0 = L.speak(sylls, **kw)
        mix.add(x, t0, "voice", gain, p=p, send=send)

    # intro: whispered ventilation word, slowly (one syllable per 8th)
    sy = []
    for bar in (4, 6):
        sy += L.word("ventilation", "high", at(bar), S16 * 2, dur=S16 * 1.8)
    say(sy, 0.55, whisper=0.9, send=0.6)

    # build: medium-priority cardiac (tresillo) 'du da de'
    sy = []
    for bar in range(8, 15):
        sy += L.word("cardiac", "medium", at(bar), S16, dur=S16 * 1.6, amp=0.6 + 0.05 * (bar - 8))
    say(sy, 0.6, send=0.35)

    # drop 1: the cardiac high-priority alarm, spoken, every bar
    sy = []
    for bar in range(16, 32):
        sy += L.word("cardiac", "high", at(bar), S16, dur=S16 * 0.85)
    say(sy, 0.85, send=0.2)

    # breakdown: the four words of the conversation, one syllable per beat, sung legato
    sy = []
    for i, cat in enumerate(CONVERSATION):
        notes = M.iec_melody(cat, "high")[:5]
        for j, (on, semi) in enumerate(notes):
            start = at(32 + i, j * 0.8)
            sy.append(L.Syllable(start, BEAT * 0.75, M.nt(semi), L.DEGREE_VOWEL[semi % 12],
                                 L.CATEGORY_CONSONANT[cat], 0.8))
    say(sy, 0.6, send=0.6, whisper=0.15, vibrato=0.008)
    # desaturation: the oxygen alarm begins once SpO2 falls under ~90 %
    sy = []
    for bar in range(36, 40):
        sy += L.word("oxygen", "medium", at(bar, 2), S16 * 1.5, dur=S16 * 1.4, amp=0.7)
    say(sy, 0.55, send=0.5)
    # rising: the temperature word over the kettle, accelerating
    sy = []
    for bar in range(40, 44):
        sy += L.word("temperature", "high", at(bar), S16, dur=S16 * 0.8, amp=0.5 + 0.15 * (bar - 40))
    say(sy, 0.55, send=0.3)
    # over the defib charge: the power-failure word
    sy = []
    for bar in (45, 46):
        sy += L.word("power", "high", at(bar), S16, dur=S16 * 0.8, amp=0.6 + 0.3 * (bar - 45))
    say(sy, 0.5, send=0.4)

    # drop 2: the machines talk to each other, one word per bar, each on the chord it spells
    sy = []
    sy_low = []
    for bar in range(48, 72):
        cat = CONVERSATION[bar % 4]
        sy += L.word(cat, "high", at(bar), S16, dur=S16 * 0.85)
        if bar >= 64:
            sy_low += L.word(cat, "high", at(bar), S16, dur=S16 * 0.85, octave=-1, amp=0.8)
    say(sy, 0.85, send=0.2)
    if sy_low:
        say(sy_low, 0.45, p=-0.2, send=0.2, brightness=0.8)

    # outro: whispered ventilation again, then silence but the vent and the heart
    sy = []
    for bar in (76, 78):
        sy += L.word("ventilation", "high", at(bar), S16 * 2, dur=S16 * 1.8)
    say(sy, 0.5, whisper=0.9, send=0.7)


def write_wav(path, stereo):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = (np.clip(stereo.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(a.tobytes())


def write_mono(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    p.add_argument("--samples", action="store_true", help="also write the machine sample pack")
    args = p.parse_args()

    if args.samples:
        for name, fn in M.CATALOGUE.items():
            write_mono(os.path.join(args.out, "samples", name + ".wav"), fn())
        for cat in L.CATEGORY_CONSONANT:
            x, _ = L.speak(L.word(cat, "high", 0.05, S16, dur=S16 * 0.85))
            write_mono(os.path.join(args.out, "samples", f"telemetric_{cat}.wav"), x)
        print("sample pack written")

    mix = Mix(N_BARS * BAR + TAIL)
    build(mix)
    out = mix.master()
    wav = os.path.join(args.out, "sinus_rhythm.wav")
    write_wav(wav, out)
    print(f"wrote {wav} ({out.shape[1] / SR:.1f} s)")
    for k, s in mix.stems.items():
        print(f"  stem {k:6s} rms {20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12):6.1f} dB  "
              f"peak {20 * np.log10(np.abs(s).max() + 1e-12):6.1f} dB")
    if shutil.which("ffmpeg"):
        mp3 = wav[:-4] + ".mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "192k",
                        "-metadata", "title=Sinus Rhythm", "-metadata", "artist=alien-music / hospital", mp3],
                       check=True)
        print(f"wrote {mp3}")


if __name__ == "__main__":
    main()
