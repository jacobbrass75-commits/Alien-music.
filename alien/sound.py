"""Sound primitives: timbres, a tunable 'ear' (roughness model), scale derivation, and a tiny synth.

Nothing here is learned from human music. Consonance comes from a physical roughness model
(Plomp-Levelt / Sethares), and the scale comes from the minima of the dissonance curve of
whatever timbre you hand it, so a weird timbre yields a weird scale and a weird "octave".
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

PHI = (1 + 5 ** 0.5) / 2


def phi_ratios(n):
    """n duration/IOI ratios spaced by sqrt(phi): irrational, so no simple meter is privileged."""
    k = np.arange(n) - (n - 1) / 2
    return (PHI ** (k / 2)).tolist()


def cents(ratio):
    return 1200 * math.log2(ratio)


# ---------------------------------------------------------------------------- ear

@dataclass
class Ear:
    """Sethares' parameterisation of Plomp-Levelt roughness.

    Human hearing is dstar=0.24, s1=0.0207, s2=18.96. s1/s2 set how the critical band scales
    with frequency; change them and you have a listener with different consonances.
    """
    dstar: float = 0.24
    s1: float = 0.0207
    s2: float = 18.96
    b1: float = 3.5
    b2: float = 5.75
    name: str = "human"

    @classmethod
    def alien(cls, seed=0):
        rng = np.random.default_rng(seed + 7919)
        return cls(s1=0.0207 * float(rng.uniform(1.8, 4.0)),
                   s2=18.96 * float(rng.uniform(0.3, 3.0)),
                   name=f"alien-{seed}")

    def roughness(self, freqs, amps):
        """freqs, amps: [..., P] tensors of partials. Returns [...] total pairwise roughness."""
        f1, f2 = freqs.unsqueeze(-1), freqs.unsqueeze(-2)
        a = torch.minimum(amps.unsqueeze(-1), amps.unsqueeze(-2))
        df = (f1 - f2).abs()
        s = self.dstar / (self.s1 * torch.minimum(f1, f2) + self.s2)
        d = a * (torch.exp(-self.b1 * s * df) - torch.exp(-self.b2 * s * df))
        return d.sum(dim=(-1, -2)) / 2


def make_ear(kind, seed=0):
    return Ear() if kind == "human" else Ear.alien(seed)


# ---------------------------------------------------------------------------- timbre

@dataclass
class Timbre:
    ratios: list
    amps: list
    decay: float = 1.5
    name: str = "harmonic"

    @classmethod
    def harmonic(cls, n=7):
        k = np.arange(1, n + 1)
        return cls(k.astype(float).tolist(), (0.88 ** (k - 1)).tolist())

    @classmethod
    def alien(cls, seed=0, n=7):
        """Stretched partials f_k = k^stretch: a consistent but non-octave spectrum."""
        rng = np.random.default_rng(seed)
        stretch = rng.uniform(1.1, 1.35)
        k = np.arange(1, n + 1)
        ratios = k ** stretch * (1 + rng.normal(0, 0.006, n))
        ratios[0] = 1.0
        amps = 0.85 ** (k - 1) * rng.uniform(0.5, 1.0, n)
        amps[0] = 1.0
        return cls(ratios.tolist(), amps.tolist(), decay=float(rng.uniform(0.8, 2.5)),
                   name=f"alien-{seed} (stretch {stretch:.3f})")


def make_timbre(kind, seed=0):
    return Timbre.harmonic() if kind == "harmonic" else Timbre.alien(seed)


# ---------------------------------------------------------------------------- scales

def dissonance_curve(timbre, ear, f0=220.0, lo=1.0, hi=3.2, n=4000):
    r = torch.linspace(lo, hi, n, dtype=torch.float64)
    base = torch.tensor(timbre.ratios, dtype=torch.float64) * f0
    a = torch.tensor(timbre.amps, dtype=torch.float64)
    freqs = torch.cat([base.expand(n, -1), r[:, None] * base], -1)
    amps = torch.cat([a, a]).expand(n, -1)
    return r.numpy(), ear.roughness(freqs, amps).numpy()


def derive_scale(timbre, ear, f0=220.0, max_notes=7, min_step_cents=60, period_range=(1.6, 3.2)):
    """Scale = most prominent minima of the dissonance curve; period = deepest minimum in
    period_range (the timbre's own 'octave'). Returns (sorted ratios starting at 1.0, period)."""
    r, d = dissonance_curve(timbre, ear, f0, hi=period_range[1])
    mins = np.where((d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:]))[0] + 1
    w = max(len(d) // 40, 1)

    def prominence(i):
        return min(d[max(i - w, 0):i].max(initial=d[i]), d[i + 1:i + w + 1].max(initial=d[i])) - d[i]

    in_range = [i for i in mins if period_range[0] <= r[i] <= period_range[1]]
    period = float(r[max(in_range, key=prominence)]) if in_range else 2.0
    chosen = [1.0]
    for i in sorted((i for i in mins if 1.0 < r[i] < period), key=prominence, reverse=True):
        if len(chosen) >= max_notes:
            break
        if all(abs(cents(r[i] / c)) >= min_step_cents for c in chosen + [period]):
            chosen.append(float(r[i]))
    return sorted(chosen), period


# ---------------------------------------------------------------------------- synth

def render_notes(notes, timbre, sr=22050, release=0.25):
    """notes: iterable of (start_s, dur_s, f0_hz, amp). Additive synthesis with the given timbre."""
    notes = list(notes)
    if not notes:
        return np.zeros(sr // 2)
    tail = release * 4
    out = np.zeros(int((max(s + d for s, d, _, _ in notes) + tail + 0.1) * sr) + 1)
    for start, dur, f0, amp in notes:
        n = int((dur + tail) * sr)
        t = np.arange(n) / sr
        gate = np.where(t < dur, 1.0, np.exp(-(t - dur) / release)) * np.minimum(t / 0.008, 1.0)
        sig = np.zeros(n)
        for k, (r, a) in enumerate(zip(timbre.ratios, timbre.amps)):
            f = f0 * r
            if f < sr * 0.45:
                sig += a * np.sin(2 * np.pi * f * t) * np.exp(-t * timbre.decay * (1 + 0.4 * k))
        i0 = int(start * sr)
        m = min(n, len(out) - i0)
        out[i0:i0 + m] += amp * (sig * gate)[:m]
    return out


@dataclass
class Drum:
    """Modal percussion: inharmonic partials + a noise burst."""
    f0: float
    ratios: list
    amps: list
    decay: float
    noise: float
    noise_decay: float

    @classmethod
    def alien(cls, seed, i, n=5):
        rng = np.random.default_rng(seed * 1000 + i + 1)
        f0 = float(np.exp(rng.uniform(math.log(50), math.log(1200))))
        ratios = np.cumsum(np.concatenate([[1.0], rng.uniform(0.3, 1.7, n - 1)]))
        amps = rng.uniform(0.2, 1.0, n)
        amps[0] = 1.0
        return cls(f0, ratios.tolist(), amps.tolist(), float(rng.uniform(5, 30)),
                   float(rng.uniform(0, 0.7)), float(rng.uniform(20, 80)))

    def render(self, sr, rng, length=0.8):
        t = np.arange(int(length * sr)) / sr
        sig = np.zeros_like(t)
        for r, a in zip(self.ratios, self.amps):
            if self.f0 * r < sr * 0.45:
                sig += a * np.sin(2 * np.pi * self.f0 * r * t) * np.exp(-t * self.decay * math.sqrt(r))
        sig += self.noise * rng.standard_normal(len(t)) * np.exp(-t * self.noise_decay)
        return sig * np.minimum(t / 0.002, 1.0)


def render_hits(hits, drums, sr=22050, seed=0):
    """hits: iterable of (time_s, drum_index, amp)."""
    hits = list(hits)
    rng = np.random.default_rng(seed)
    samples = [d.render(sr, rng) for d in drums]
    if not hits:
        return np.zeros(sr // 2)
    out = np.zeros(int((max(h[0] for h in hits) + 1.0) * sr))
    for time, k, amp in hits:
        s = samples[k]
        i0 = int(time * sr)
        m = min(len(s), len(out) - i0)
        out[i0:i0 + m] += amp * s[:m]
    return out

