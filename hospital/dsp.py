"""Small DSP toolkit: oscillators, envelopes, filters (static and time-varying), reverb, panning."""
import numpy as np
from scipy import signal

SR = 44100
RNG = np.random.default_rng(60)  # 60, for the mains hum the whole track is tuned to


def n_(sec):
    return max(1, int(round(sec * SR)))


def tt(sec):
    return np.arange(n_(sec)) / SR


def noise(sec):
    return RNG.standard_normal(n_(sec))


def _freq_array(freq, sec):
    n = n_(sec)
    if np.isscalar(freq):
        return np.full(n, float(freq))
    f = np.asarray(freq, float)
    return f[:n] if len(f) >= n else np.pad(f, (0, n - len(f)), mode="edge")


def harmonics(freq, sec, amps, ratios=None, vibrato=0.0, vib_rate=5.5):
    """Additive tone. Partials above ~0.45*SR are dropped (no aliasing)."""
    f = _freq_array(freq, sec)
    if vibrato:
        f = f * (1 + vibrato * np.sin(2 * np.pi * vib_rate * np.arange(len(f)) / SR))
    ph = 2 * np.pi * np.cumsum(f) / SR
    ratios = np.arange(1, len(amps) + 1) if ratios is None else ratios
    out = np.zeros(len(f))
    for r, a in zip(ratios, amps):
        out += a * np.sin(r * ph) * (f * r < SR * 0.45)
    return out


def sine(freq, sec):
    return harmonics(freq, sec, [1.0])


# ---------- envelopes ----------
def perc(n, attack, decay):
    """Linear attack, exponential decay with time constant `decay` seconds."""
    t = np.arange(n) / SR
    a = max(attack, 1e-4)
    return np.where(t < a, t / a, np.exp(-(t - a) / max(decay, 1e-4)))


def trap(n, rise, fall):
    """Trapezoid: linear rise, flat top, linear fall (IEC 60601-1-8 pulse shape)."""
    e = np.ones(n)
    r, f = min(n // 2, n_(rise)), min(n // 2, n_(fall))
    if r:
        e[:r] = np.linspace(0, 1, r)
    if f:
        e[-f:] = np.minimum(e[-f:], np.linspace(1, 0, f))
    return e


def ar(n, attack, release):
    return trap(n, attack, release)


# ---------- filters ----------
def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, min(f, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos"), x)


def _biquad(kind, f, q):
    w = 2 * np.pi * min(max(f, 20.0), SR * 0.45) / SR
    c, s = np.cos(w), np.sin(w)
    al = s / (2 * q)
    if kind == "bp":  # constant 0 dB peak gain
        b = [al, 0.0, -al]
    elif kind == "lp":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
    else:
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    a = [1 + al, -2 * c, 1 - al]
    return np.array(b) / a[0], np.array(a) / a[0]


def reson(x, f, q):
    """A single resonant mode (a struck panel, a gradient coil, a mouth formant)."""
    b, a = _biquad("bp", f, q)
    return signal.lfilter(b, a, x)


def tv_filter(x, freqs, q=0.7, kind="lp", bw=None, block=64):
    """Time-varying biquad. freqs: per-sample cutoff/centre. If bw is given, Q = f / bw (formants)."""
    freqs = np.broadcast_to(np.asarray(freqs, float), x.shape) if np.ndim(freqs) else np.full(len(x), freqs)
    out = np.empty_like(x)
    zi = np.zeros(2)
    for i in range(0, len(x), block):
        f = freqs[i]
        b, a = _biquad(kind, f, (f / bw) if bw else q)
        out[i:i + block], zi = signal.lfilter(b, a, x[i:i + block], zi=zi)
    return out


# ---------- space ----------
def pan(x, p):
    """Equal-power pan, p in [-1, 1]. Returns (2, n)."""
    th = (p + 1) * np.pi / 4
    return np.vstack([x * np.cos(th), x * np.sin(th)])


def reverb_ir(sec=2.8, predelay=0.02, bright=7000.0, seed=1):
    """Exponentially decaying decorrelated noise, darkening as it decays: a big tiled room (an ICU)."""
    rng = np.random.default_rng(seed)
    n = n_(sec)
    t = np.arange(n) / SR
    chans = []
    for _ in range(2):
        x = rng.standard_normal(n) * np.exp(-6.9 * t / sec)
        early = lp(x, bright)
        late = lp(x, bright / 4)
        mix = np.clip(t / sec * 2, 0, 1)
        x = early * (1 - mix) + late * mix
        x = np.concatenate([np.zeros(n_(predelay)), x])
        chans.append(x / np.sqrt(np.sum(x ** 2)))
    return np.vstack(chans)


def convolve_stereo(x, ir):
    return np.vstack([signal.fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
