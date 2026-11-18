"""Telemetric: a resonant language spoken by hospital machines.

Source-filter speech, built from the ward:
  throat  (source)  = a buzzing harmonic series, like the MRI's EPI readout tone (voiced)
                      or ventilator airflow (whispered)
  melody  (pitch)   = the IEC 60601-1-8 alarm melodies, note for note
  mouth   (filter)  = four high-Q resonators (formants) that glide between vowels
  consonants        = the machines' own transients: MRI knock, pump click, heartbeat thump, O2 hiss...

Grammar: every alarm melody is a word. Each pulse is a syllable:
  consonant = the alarm's category (which machine is talking)
  vowel     = the pulse's pitch (scale degree)
so the IEC cardiac high-priority alarm  c e g . g C  is pronounced  du da de . de du.
"""
from dataclasses import dataclass

import numpy as np

from . import machines as M
from .dsp import SR, ar, bp, hp, n_, noise, tv_filter

# F1..F4 (Hz). Bandwidths are narrow on purpose: this is a resonant language.
VOWELS = {
    "u":  (300, 870, 2240, 3300),
    "o":  (500, 900, 2500, 3400),
    "a":  (730, 1090, 2440, 3400),
    "ae": (660, 1720, 2410, 3400),
    "e":  (530, 1840, 2480, 3500),
    "i":  (270, 2290, 3010, 3700),
}
BANDWIDTHS = (70, 100, 140, 200)
GAINS = (1.0, 1.1, 0.8, 0.45)

# pitch class -> vowel (brighter vowel for higher degree; the octave folds back to u)
DEGREE_VOWEL = {0: "u", 2: "o", 4: "a", 5: "ae", 6: "ae", 7: "e", 9: "i", 11: "i"}

# alarm category -> consonant -> which machine makes it
CATEGORY_CONSONANT = {
    "general": "k",      # MRI gradient knock
    "cardiac": "d",      # heartbeat thump
    "perfusion": "w",    # pump whir glide
    "ventilation": "h",  # ventilator breath
    "oxygen": "s",       # O2 flowmeter hiss
    "temperature": "f",  # kettle steam
    "drug": "t",         # infusion pump click
    "power": "z",        # defibrillator buzz
}
# formant positions the mouth starts from for each consonant before gliding to the vowel
LOCUS = {
    "d": (250, 1800, 2700, 3500), "t": (250, 1800, 2800, 3500), "k": (300, 2100, 2600, 3500),
    "s": (300, 1700, 2700, 3600), "z": (300, 1700, 2700, 3600), "f": (300, 1100, 2400, 3400),
    "w": (300, 650, 2200, 3300), "h": None,
}
NOISY = {"h": 0.09, "s": 0.06, "z": 0.06, "f": 0.07}


@dataclass
class Syllable:
    start: float
    dur: float
    f0: float
    vowel: str
    cons: str = ""
    amp: float = 1.0


def word(category, priority, t0, step, f_base=M.C4, dur=None, amp=1.0, octave=0):
    """The syllables of one alarm, starting at t0, with `step` seconds per 16th-grid unit."""
    cons = CATEGORY_CONSONANT[category]
    out = []
    for on, semi in M.iec_melody(category, priority):
        out.append(Syllable(t0 + on * step, dur or step * 0.9, f_base * 2 ** ((semi + 12 * octave) / 12),
                            DEGREE_VOWEL[semi % 12], cons, amp))
    return out


def spell(category, priority="high"):
    """Romanised spelling, e.g. 'du da de . de du . . du da de . de du'."""
    cons = CATEGORY_CONSONANT[category]
    notes = M.iec_melody(category, priority)
    slots = ["."] * (notes[-1][0] + 1)
    for on, semi in notes:
        slots[on] = cons + DEGREE_VOWEL[semi % 12]
    return " ".join(slots)


def _consonant(c):
    if c == "k":
        return hp(M.mri_knock(1440, 0.06), 1500) * 0.6
    if c == "t":
        return M.pump_click() * 0.9
    if c == "d":
        x = M._thump(90, 0.06, 0.02, click=0.8) * 0.5
        x[: n_(0.02)] += bp(noise(0.02), 1200, 4000) * 0.15 * np.exp(-np.arange(n_(0.02)) / 200)
        return x
    if c == "s":
        return M.o2_hiss(0.07, 4500) * 0.25
    if c == "z":
        x = M.o2_hiss(0.07, 3500) * 0.2
        t = np.arange(len(x)) / SR
        return x * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 120 * t)))
    if c == "f":
        return bp(noise(0.06), 1500, 8000) * 0.12 * ar(n_(0.06), 0.01, 0.03)
    return np.zeros(1)


def speak(sylls, whisper=0.0, vibrato=0.004, glide=0.035, brightness=1.0):
    """Render syllables. Returns (mono audio, start time in seconds)."""
    sylls = sorted(sylls, key=lambda s: s.start)
    t0 = sylls[0].start - 0.05
    t1 = max(s.start + s.dur for s in sylls) + 0.4
    n = n_(t1 - t0)

    amp = np.zeros(n)
    asp = np.zeros(n)
    fx, fp = [], []
    form_x, form_p = [], [[] for _ in range(4)]
    for s in sylls:
        i0 = n_(s.start - t0)
        i1 = min(n, i0 + n_(s.dur))
        amp[i0:i1] = np.maximum(amp[i0:i1], ar(i1 - i0, 0.008, min(0.05, s.dur * 0.35)) * s.amp)
        fx += [i0, i1 - 1]
        fp += [s.f0, s.f0]
        v = VOWELS[s.vowel]
        loc = LOCUS.get(s.cons) or v
        g = n_(glide)
        form_x += [i0, i0 + g]
        for k in range(4):
            form_p[k] += [loc[k], v[k]]
        if s.cons in NOISY:  # aspiration / frication goes through the mouth too
            m = n_(NOISY[s.cons])
            seg = slice(max(0, i0 - m // 2), min(n, i0 + m // 2))
            asp[seg] = np.maximum(asp[seg], np.hanning(seg.stop - seg.start) * s.amp)

    idx = np.arange(n)
    f0 = np.interp(idx, fx, fp)
    f0 *= 1 + vibrato * np.sin(2 * np.pi * 5.2 * idx / SR)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    voiced = np.zeros(n)
    for k in range(1, 40):
        voiced += np.sin(k * ph) / k ** 0.75 * (f0 * k < 7000 * brightness)
    air = bp(noise(n / SR)[:n], 150, 8000)
    src = (voiced * (1 - whisper) + air * whisper * 1.6) * amp + air * asp * 0.9

    out = np.zeros(n)
    for k in range(4):
        fk = np.interp(idx, form_x, form_p[k])
        out += GAINS[k] * tv_filter(src, fk, bw=BANDWIDTHS[k], kind="bp")
    # a little of the dry throat keeps the pitch legible
    out += 0.015 * src

    for s in sylls:
        if s.cons and s.cons not in ("h", "w"):
            c = _consonant(s.cons) * s.amp
            i = max(0, n_(s.start - t0) - n_(0.01))
            out[i:i + len(c)] += c[: max(0, n - i)]
    return out, t0
