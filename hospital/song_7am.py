"""'Shift Change': the 7 AM version. Darker, harder, much bigger bass.

  python -m hospital.song_7am            # -> hospital/out/shift_change.wav (+ .mp3)

At 7 AM the ward wakes up. Shift-change noise has been measured above 100 dBA, and patients rate
mornings as the worst time of day. This track is built from the machines that make it that way:
  * piezo buzzers (bed-exit and IV-pump occlusion alarms): square waves at a 2-4 kHz disc resonance
  * the automatic blood-pressure cuff on morning vitals: a mains-frequency diaphragm pump grinding up,
    then ~10 valve-snap deflation steps
  * the pneumatic tube station: a carrier rushing up the pipe and slamming in
  * the floor scrubber doing the corridor before rounds
  * the portable X-ray: generator whine, then the exposure BEEP

140 BPM, 88 bars, C minor-ish (C C Ab G). The mains hum is the bass again, now a rolling 16th-note
bass: one note on each 16th that isn't a kick, wave-shaped and filtered, with a clean sub under it.
"""
import os
import shutil
import subprocess

import numpy as np

from . import language as L
from . import machines as M
from .dsp import SR, hp, lp, n_, noise, tv_filter
from .song import Mix, write_wav

BPM = 140
BEAT = 60 / BPM
BAR = 4 * BEAT
S16 = BEAT / 4
N_BARS = 88
TAIL = 3.5
ROOTS = [-24, -24, -28, -29]  # C C Ab G, bass register (60, 60, 47.6, 44.9 Hz)
CONVERSATION = ["cardiac", "ventilation", "drug", "oxygen"]
SECTIONS = {"intro": (0, 16), "build": (16, 24), "drop1": (24, 40), "breakdown": (40, 52),
            "flatline": (52, 56), "drop2": (56, 80), "outro": (80, 88)}


def at(bar, beat=0.0, s16=0.0):
    return bar * BAR + beat * BEAT + s16 * S16


def in_(name, bar):
    a, b = SECTIONS[name]
    return a <= bar < b


class HardMix(Mix):
    DUCK = {"drums": 0.0, "bass": 0.6, "pad": 0.7, "voice": 0.35, "fx": 0.35, "perc": 0.2, "kick": 0.0}


def build(mix):
    rng = np.random.default_rng(7)
    drop = lambda bar: in_("drop1", bar) or in_("drop2", bar)  # noqa: E731

    def say(sylls, gain, p=0.0, send=0.2, drive=2.5, **kw):
        x, t0 = L.speak(sylls, **kw)
        x = np.tanh(drive * x / (np.abs(x).max() + 1e-9)) / np.tanh(drive)
        mix.add(x, t0, "voice", gain, p=p, send=send)

    # ------------------------------------------------------------ the corridor at 7 AM
    scrub = M.floor_scrubber(at(16))
    k = np.arange(len(scrub)) / len(scrub)
    scrub = tv_filter(scrub, 250 * (6000 / 250) ** np.clip(k * 1.6, 0, 1), q=0.7, kind="lp") * np.clip(k * 2.5, 0, 1)
    mix.add(scrub, 0.0, "fx", 0.22, p=-0.25, send=0.15)
    sc2 = M.floor_scrubber(at(52) - at(40))
    sc2 *= np.minimum(1, np.linspace(0, 3, len(sc2))) * np.minimum(1, np.linspace(3, 0, len(sc2)))
    mix.add(sc2, at(40), "fx", 0.16, p=0.3, send=0.1)
    mix.add(M.mains_hum(60.0, at(16), buzz=0.4) * np.linspace(0.3, 1, n_(at(16))), 0.0, "bass", 0.18)
    mix.add(M.nurse_chime((7, 4, 0)), at(0), "fx", 0.2, p=0.4, send=0.6)

    # morning vitals: the BP cuff, inflating then stepping down on the beat
    for bar in (2, 10, 40):
        mix.add(M.nibp_inflate(2 * BAR), at(bar), "fx", 0.2, p=0.35, send=0.1)
        for i in range(10):
            mix.add(M.nibp_step(), at(bar + 2, i), "perc", 0.35, p=0.35, send=0.15)

    # an IV pump down the hall, then closer
    for bar in range(4, 16, 2):
        d = (bar - 4) / 12
        for j in range(3):
            x = lp(M.piezo(3840, 0.09), 1500 + 9000 * d)
            mix.add(x, at(bar, 1, j * 2), "fx", 0.06 + 0.12 * d, p=-0.5, send=0.5)

    # ------------------------------------------------------------ heart monitor
    for bar in range(N_BARS):
        if in_("intro", bar) or (in_("outro", bar) and bar >= 84):
            beats = [0, 2]
        elif in_("build", bar) or drop(bar) or (in_("outro", bar) and bar < 84):
            beats = [0, 1, 2, 3]
        elif in_("breakdown", bar) and bar >= 49:
            beats = [i / 2 for i in range(8)] if bar < 51 else [i / 4 for i in range(16)]
        else:
            beats = []
        for b in beats:
            mix.add(M.qrs_beep(), at(bar, b), "perc", 0.08 if drop(bar) else 0.14, p=0.2, send=0.1)
    for kk in range(32):  # desaturation in the breakdown
        spo2 = int(round(99 - 23 * kk / 31))
        mix.add(M.pulseox_beep(spo2, 0.1), at(44 + kk // 4, kk % 4), "perc", 0.2, p=0.15, send=0.35)

    # ------------------------------------------------------------ drums
    def kick(t, gain=1.0, cutoff=None):
        x = M.hard_kick()
        if cutoff:
            x = lp(x, cutoff)
        mix.add(x, t, "kick", gain * 0.75)
        mix.kicks.append(t)

    for bar in range(N_BARS):
        if drop(bar) or in_("build", bar) and bar < 23 or in_("outro", bar) and bar < 84:
            for b in range(4):
                kick(at(bar, b))
            if drop(bar) or in_("build", bar):
                for b in (1, 3):
                    mix.add(M.zap(), at(bar, b), "drums", 1.1, send=0.12)
            if drop(bar):
                for s in range(16):
                    vel = 0.5 if s % 4 == 2 else 0.22 if s % 2 else 0.3
                    mix.add(M.pump_click(), at(bar, 0, s), "perc", vel * 0.7, p=0.35 if s % 2 else -0.35)
                for b in range(4):  # the piezo takes the off-beat: the most abrasive hi-hat in the building
                    mix.add(M.piezo(3840, 0.045), at(bar, b + 0.5), "perc", 0.16, p=0.15)
                for s, res in zip([3, 6, 11, 14], [720, 960, 720, 1080]):
                    mix.add(np.tanh(3 * M.mri_knock(res)), at(bar, 0, s), "perc", 0.22, p=-0.4 if s < 8 else 0.4)
        elif in_("build", bar) and bar == 23:
            kick(at(bar))
        elif in_("intro", bar) and bar >= 8:
            for b in range(4):
                kick(at(bar, b), 0.55, cutoff=220 + 60 * (bar - 8))
    # snare roll into drop 1
    for i, s in enumerate(range(0, 16, 2)):
        mix.add(M.zap(), at(22, 0, s), "drums", 0.4 + 0.05 * i)
    for s in range(12):
        mix.add(M.zap(), at(23, 0, s), "drums", 0.6 + 0.04 * s)
    # MRI roll in the breakdown
    for i in range(48):  # 16ths through bar 50, 32nds through bar 51
        p = i / 48
        t = at(50, 0, i) if i < 16 else at(51, 0, (i - 16) / 2)
        mix.add(M.mri_knock(720 * 2 ** p), t, "perc", 0.2 + 0.3 * p)

    # ------------------------------------------------------------ bass
    for bar in range(N_BARS):
        f = M.nt(ROOTS[bar % 4])
        rolling = drop(bar) or (in_("build", bar) and bar < 23) or (in_("outro", bar) and bar < 82)
        if rolling:
            if in_("build", bar):
                cut = 300 + 150 * (bar - 16)
            elif in_("drop2", bar) and bar >= 72:
                cut = 2800
            else:
                cut = 2000
            for b in range(4):
                for s in (1, 2, 3):
                    mix.add(M.rolling_bass(f, S16 * 0.95, cutoff=cut), at(bar, b, s), "bass", 0.7)
        if (in_("intro", bar) and bar >= 8) or in_("breakdown", bar) or in_("flatline", bar) and bar < 53 \
                or (in_("outro", bar) and bar >= 82):
            x = M.reese(f, BAR + 0.05, cutoff=180 + (40 * (bar - 8) if bar < 16 else 500))
            tb = np.arange(len(x)) / SR
            x *= 0.75 + 0.25 * np.sin(2 * np.pi * tb / (BEAT * 2))  # slow throb
            mix.add(x, at(bar), "bass", 0.5)

    # dark power-chord drone made of alarm harmonics (no thirds)
    for bar in range(8, N_BARS):
        if in_("flatline", bar):
            continue
        r = ROOTS[bar % 4]
        g = 0.13 if drop(bar) else 0.2
        for semi in (r + 24, r + 31, r + 36):
            x = M.alarm_pad_tone(M.nt(semi), BAR, detune_cents=12, attack=0.2, release=0.4)
            mix.add(lp(x, 900 if drop(bar) else 1400), at(bar), "pad", g, p=rng.uniform(-0.6, 0.6), send=0.3)

    # ------------------------------------------------------------ abrasive alarms
    # bed-exit siren: two-tone piezo, 16ths
    for bar0, bars in ((20, 4), (64, 2), (68, 2), (76, 4)):
        for kk in range(bars * 16):
            if bar0 in (64, 68) and kk % 16 < 8:
                continue
            f = 3840 if kk % 2 == 0 else 2880
            gain = 0.05 + 0.12 * kk / (bars * 16) if bar0 == 20 else 0.14
            mix.add(M.piezo(f, S16 * 0.8), at(bar0, 0, kk), "fx", gain, p=0.5 if kk % 2 else -0.5, send=0.1)
    # infusion occlusion: triple fat beeps
    for bar in list(range(32, 40, 2)) + list(range(48, 52)) + list(range(58, 80, 2)):
        for j in range(3):
            mix.add(M.occlusion_beep(1920, S16 * 1.6), at(bar, 2.5) + j * S16 * 2, "fx", 0.13, p=0.25, send=0.15)
    # portable X-ray: generator whine as the riser, exposure BEEP on the drops
    mix.add(M.xray_charge(4 * BAR), at(20), "fx", 0.22, send=0.25)
    mix.add(M.xray_charge(4 * BAR), at(76), "fx", 0.18, send=0.25)
    for bar in (24, 56, 64, 72):
        mix.add(M.occlusion_beep(1920, BEAT * 1.5), at(bar), "fx", 0.16, send=0.3)
    # pneumatic tube: the carrier rushes up during the bar before, slams on the downbeat
    for bar in (24, 32, 56, 64, 72, 88):
        x = M.pneumatic_tube(whoosh=BAR)
        mix.add(x, at(bar - 1), "drums", 0.75, send=0.35)
    # EPI stabs
    for bar in list(range(32, 40)) + list(range(64, 80)):
        for s in ((7, 15) if bar < 72 else (2, 7, 10, 15)):
            mix.add(lp(M.epi_buzz(0.05), 7000), at(bar, 0, s), "perc", 0.06, p=0.5 if s % 2 else -0.5)

    # ------------------------------------------------------------ flatline -> charge -> shock
    mix.add(M.flatline(BAR), at(52), "fx", 0.22, send=0.5)
    mix.add(M.defib_charge(at(55, 2) - at(53)), at(53), "fx", 0.25, send=0.3)
    mix.add(M.flatline(BEAT * 1.4, f=1920.0) * 0.6, at(55, 2), "fx", 0.16)
    mix.add(M.defib_shock(), at(56), "drums", 0.9, send=0.4)

    # ------------------------------------------------------------ Telemetric, darker: an octave down, driven
    sy = []
    for bar in (8, 10, 12, 14):
        sy += L.word("ventilation", "high", at(bar), S16 * 2, dur=S16 * 1.8, octave=-1)
    say(sy, 0.45, whisper=0.85, send=0.5, drive=1.5)
    sy = []
    for bar in range(16, 23):
        sy += L.word("cardiac", "medium", at(bar), S16, dur=S16 * 1.6, octave=-1, amp=0.6 + 0.05 * (bar - 16))
    say(sy, 0.55)
    for octv, g in ((-1, 0.4), (0, 0.18)):
        sy = []
        for bar in range(24, 40):
            sy += L.word("cardiac", "high", at(bar), S16, dur=S16 * 0.85, octave=octv)
        say(sy, g, p=0.0 if octv else 0.2, drive=3.0 if octv else 2.0)
    sy = []
    for i, cat in enumerate(CONVERSATION):
        for j, (on, semi) in enumerate(M.iec_melody(cat, "high")[:5]):
            sy.append(L.Syllable(at(40 + i, j * 0.8), BEAT * 0.75, M.nt(semi - 12), L.DEGREE_VOWEL[semi % 12],
                                 L.CATEGORY_CONSONANT[cat], 0.8))
    say(sy, 0.5, send=0.6, whisper=0.2, vibrato=0.008, drive=1.8)
    sy = []
    for bar in range(46, 50):
        sy += L.word("oxygen", "medium", at(bar, 2), S16 * 1.5, dur=S16 * 1.4, amp=0.7, octave=-1)
    say(sy, 0.5, send=0.5)
    sy = []
    for bar in (53, 54):
        sy += L.word("power", "high", at(bar), S16, dur=S16 * 0.8, octave=-1, amp=0.6 + 0.3 * (bar - 53))
    say(sy, 0.5, send=0.4, drive=3.0)
    for octv, g in ((-1, 0.5), (0, 0.22)):
        sy = []
        for bar in range(56, 80):
            sy += L.word(CONVERSATION[bar % 4], "high", at(bar), S16, dur=S16 * 0.85, octave=octv)
        say(sy, g, p=0.0 if octv else -0.2, drive=3.5 if octv else 2.0)
    sy = []
    for bar in (84, 86):
        sy += L.word("ventilation", "high", at(bar), S16 * 2, dur=S16 * 1.8, octave=-1)
    say(sy, 0.45, whisper=0.85, send=0.6, drive=1.5)


def main(out_dir=os.path.join(os.path.dirname(__file__), "out")):
    mix = HardMix(N_BARS * BAR + TAIL)
    build(mix)
    out = mix.master(drive=2.4, wet_gain=0.4, ceiling_db=-0.5)
    wav = os.path.join(out_dir, "shift_change.wav")
    write_wav(wav, out)
    print(f"wrote {wav} ({out.shape[1] / SR:.1f} s)")
    if shutil.which("ffmpeg"):
        mp3 = wav[:-4] + ".mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "192k",
                        "-metadata", "title=Shift Change", "-metadata", "artist=alien-music / hospital", mp3],
                       check=True)
        print(f"wrote {mp3}")
    return mix


if __name__ == "__main__":
    main()
