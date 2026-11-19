"""Hospital machines, synthesised from their published specs (see hospital/README.md for sources).

Everything is tuned to the mains: C = 240 Hz = 4 x 60 Hz, so the wall-socket hum is the bass note C.
Two lucky coincidences make that tuning feel inevitable:
  * the 3 T gradient-coil resonance measured near 730 Hz lands on G (720 Hz = 12 x 60)
  * the EPI readout tone measured near 1.9 kHz lands on C (1920 Hz = 32 x 60)
"""
import numpy as np

from .dsp import SR, ar, bp, harmonics, hp, lp, n_, noise, perc, reson, sine, trap, tv_filter

C4 = 240.0  # 'c' in IEC 60601-1-8 Annex F notation; 'C' (upper case) is the octave above


def nt(semis, base=C4):
    return base * 2 ** (semis / 12)


# ---------------------------------------------------------------- patient monitor
def qrs_beep(f=720.0, sec=0.085):
    """The beep the monitor makes on every detected QRS complex (one per heartbeat)."""
    x = harmonics(f, sec, [1.0, 0.08, 0.15])
    return x * trap(len(x), 0.003, 0.02)


def spo2_freq(spo2, f100=720.0):
    """Pulse-oximeter pitch. Log mapping from the Philips-style patent: the pitch halves for every
    24 points of saturation lost, i.e. 1 % SpO2 = one quarter-tone. (The patent uses 662 Hz at 100 %;
    we put 100 % on G = 720 Hz so it sits in the track's key.)"""
    return f100 * 2 ** (-(100 - spo2) / 24)


def pulseox_beep(spo2, sec=0.1):
    return qrs_beep(spo2_freq(spo2), sec)


def flatline(sec, f=960.0):
    """Asystole: one continuous tone. 960 Hz = 16 x 60."""
    x = harmonics(f, sec, [1.0, 0.1, 0.2, 0.05])
    return x * trap(len(x), 0.005, 0.05)


# ---------------------------------------------------------------- IEC 60601-1-8 alarms
# Annex F 'melodic' alarms (2006 edition). Semitones above c.  (medium, high-priority tail)
IEC_MELODIES = {
    "general":     ([0, 0, 0],    [0, 0]),
    "cardiac":     ([0, 4, 7],    [7, 12]),
    "perfusion":   ([0, 6, 0],    [0, 6]),
    "ventilation": ([0, 9, 5],    [9, 5]),
    "oxygen":      ([12, 11, 9],  [7, 5]),
    "temperature": ([0, 2, 4],    [5, 7]),
    "drug":        ([12, 2, 7],   [12, 2]),
    "power":       ([12, 0, 0],   [12, 0]),
}
# High priority: 10 pulses as 3+2, gap, 3+2. On a 16th grid that is exactly one 4/4 bar:
#   x x x . x x . . x x x . x x . .
HIGH_ONSETS = [0, 1, 2, 4, 5, 8, 9, 10, 12, 13]
# Medium priority: 3 slower pulses, spaced 3 sixteenths: a tresillo.
MEDIUM_ONSETS = [0, 3, 6]


def iec_melody(category, priority="high"):
    med, tail = IEC_MELODIES[category]
    if priority == "medium":
        return list(zip(MEDIUM_ONSETS, med))
    five = med + tail
    return list(zip(HIGH_ONSETS, five + five))


def iec_pulse(f0, td=0.11, amps=(1.0, 0.8, 0.62, 0.48, 0.36)):
    """One alarm pulse: fundamental in 150-1000 Hz plus >= 4 harmonics (300-4000 Hz),
    rise and fall ~15 % of the pulse duration td."""
    x = harmonics(f0, td, amps)
    return x * trap(len(x), 0.15 * td, 0.15 * td)


# ---------------------------------------------------------------- IEC 60601-1-8:2020 auditory icons
def _thump(f, sec, decay, click=0.3):
    t = np.arange(n_(sec)) / SR
    fr = f * (1 + 0.8 * np.exp(-t / 0.015))
    body = sine(fr, sec) * perc(len(t), 0.002, decay)
    c = lp(noise(sec), 1800) * perc(len(t), 0.0005, 0.006) * click
    return body + c


def heartbeat(gap=0.117, lub=52.0, dub=74.0):
    """'Lup-dup' cardiovascular icon. gap 0.117 s = one 16th at 128 BPM."""
    x = np.zeros(n_(gap + 0.4))
    a = _thump(lub, 0.35, 0.07)
    b = _thump(dub, 0.3, 0.05, click=0.5) * 0.65
    x[: len(a)] += a
    i = n_(gap)
    x[i:i + len(b)] += b[: len(x) - i]
    return x


def kick():
    """A dance kick with the heartbeat's 'lub' inside it."""
    sec = 0.45
    t = np.arange(n_(sec)) / SR
    f = 44 + 140 * np.exp(-t / 0.035)
    body = sine(f, sec) * perc(len(t), 0.001, 0.16)
    click = hp(noise(sec), 2500) * perc(len(t), 0.0003, 0.003) * 0.25
    lub = _thump(52, sec, 0.07) * 0.6
    return np.tanh(1.6 * (body + click + lub)) / np.tanh(1.6)


def ventilator_breath(inhale=1.0, pause=0.5, exhale=2.25):
    """Ventilation icon (2020): ~1 s inhale (like white noise), 0.5 s gap, slow exhale.
    Scaled so one breath is two bars at 128 BPM: 16 breaths/min, I:E about 1:2, as on an ICU vent."""
    sec = inhale + pause + exhale
    n = n_(sec)
    t = np.arange(n) / SR
    src = noise(sec)
    centre = np.where(t < inhale, 500 + 1100 * (t / inhale) ** 1.5,
                      np.maximum(320, 1300 - 900 * np.clip((t - inhale - pause) / exhale, 0, 1)))
    y = tv_filter(src, centre, q=1.4, kind="bp")
    amp = np.where(t < inhale, (t / inhale) ** 1.2,
                   np.where(t < inhale + pause, 0.0,
                            0.75 * np.exp(-np.clip(t - inhale - pause, 0, None) / (exhale / 2.5))
                            * np.clip((t - inhale - pause) / 0.15, 0, 1)))
    y *= amp
    # valve clicks at the start of each phase, piston thump under the inhale
    for at in (0.0, inhale, inhale + pause):
        c = valve_click()
        i = n_(at)
        y[i:i + len(c)] += c[: n - i] * 0.6
    th = _thump(70, 0.3, 0.08, click=0.1) * 0.5
    y[: len(th)] += th
    return y


def valve_click():
    sec = 0.03
    x = noise(sec) * perc(n_(sec), 0.0003, 0.002)
    return reson(x, 1500, 6) * 2 + hp(x, 4000) * 0.3


# ---------------------------------------------------------------- MRI
def mri_knock(res=720.0, sec=0.14):
    """Gradient-coil knock: an impulse exciting the coil's modes (~730 Hz, 3-4 kHz, 5.5-6.5 kHz)."""
    exc = np.zeros(n_(sec))
    exc[:n_(0.002)] = noise(0.002)
    exc[0] = 3.0
    y = (reson(exc, res, 22) * 1.0 + reson(exc, 3500, 9) * 0.45 + reson(exc, 6000, 7) * 0.3)
    y += _thump(110, sec, 0.03, click=0.0) * 0.25
    return y * perc(len(y), 0.0005, 0.05)


def epi_buzz(sec, f0=1920.0):
    """EPI readout tone: fundamental ~1.9 kHz with harmonics at 3.8, 5.8, 7.7, 9.6 kHz."""
    x = harmonics(f0, sec, [1.0, 0.55, 0.4, 0.25, 0.18])
    x = np.tanh(2.0 * x)
    return x * trap(len(x), 0.002, 0.006)


# ---------------------------------------------------------------- infusion pump
def pump_click():
    """Peristaltic / stepper infusion pump: a dry mechanical tick."""
    sec = 0.025
    x = noise(sec) * perc(n_(sec), 0.0002, 0.0018)
    return hp(x, 3500) + reson(x, 2600, 14) * 0.8


def pump_whir(sec, step_rate=320.0):
    """Stepper motor running: a buzzy tone at the step rate."""
    x = harmonics(step_rate, sec, [0.6, 1.0, 0.3, 0.5, 0.2, 0.25])
    x += bp(noise(sec), 2000, 6000) * 0.2
    return x * trap(len(x), 0.03, 0.05)


# ---------------------------------------------------------------- defibrillator
def defib_charge(sec, f_lo=520.0, f_hi=3900.0):
    """Charging whine: the inverter's pitch tracks capacitor voltage, rising and leveling off."""
    t = np.arange(n_(sec)) / SR
    k = 2.2
    v = (1 - np.exp(-k * t / sec)) / (1 - np.exp(-k))
    f = f_lo + (f_hi - f_lo) * v
    x = harmonics(f, sec, [1.0, 0.0, 0.25]) + 0.3 * harmonics(f / 2, sec, [1.0])
    x += bp(noise(sec), 3000, 9000) * 0.08
    return x * (0.25 + 0.75 * v) * trap(len(t), 0.05, 0.01)


def defib_shock():
    """Discharge: a body thump, a crack, and a short electrical buzz."""
    sec = 1.2
    t = np.arange(n_(sec)) / SR
    thump = sine(30 + 80 * np.exp(-t / 0.04), sec) * perc(len(t), 0.001, 0.35)
    crack = bp(noise(sec), 200, 12000) * perc(len(t), 0.0005, 0.03)
    buzz = bp(noise(sec), 500, 6000) * (np.sign(np.sin(2 * np.pi * 120 * t)) * 0.5 + 0.5) * perc(len(t), 0.001, 0.08)
    return np.tanh(1.5 * (thump + crack * 0.8 + buzz * 0.5))


def zap():
    """Snare: a small defib zap (crack + 120 Hz-gated noise + body)."""
    sec = 0.3
    t = np.arange(n_(sec)) / SR
    crack = bp(noise(sec), 1500, 9000) * perc(len(t), 0.0005, 0.07)
    gate = np.sign(np.sin(2 * np.pi * 120 * t)) * 0.5 + 0.5
    buzz = bp(noise(sec), 800, 5000) * gate * perc(len(t), 0.001, 0.045)
    body = sine(190 + 60 * np.exp(-t / 0.01), sec) * perc(len(t), 0.001, 0.05)
    return crack * 0.9 + buzz * 0.5 + body * 0.6


# ---------------------------------------------------------------- respiratory / oxygen
def o2_hiss(sec, lo=5000.0):
    """Oxygen flowmeter hiss."""
    x = hp(noise(sec), lo, order=4)
    return x * trap(len(x), 0.003, min(sec / 2, 0.06))


def bubble(f0=600.0, sec=0.05):
    """Humidifier bubble: a resonant pop whose pitch rises as the bubble reaches the surface."""
    t = np.arange(n_(sec)) / SR
    f = f0 * (1 + 1.8 * t / sec)
    return sine(f, sec) * perc(len(t), 0.0008, sec / 4)


def kettle(sec, f_lo=1300.0, f_hi=2150.0):
    """Temperature icon (2020): a whistling kettle coming to the boil."""
    t = np.arange(n_(sec)) / SR
    p = (t / sec) ** 1.6
    f = f_lo + (f_hi - f_lo) * p + 25 * np.sin(2 * np.pi * 6.3 * t) * p
    tone = sine(f, sec) * np.clip(p * 1.5 - 0.15, 0, 1)
    turb = tv_filter(noise(sec), f, q=6, kind="bp") * (0.25 + 0.5 * p)
    return (tone * 0.7 + turb) * trap(len(t), 0.2, 0.02)


# ---------------------------------------------------------------- the ward
def nurse_chime(notes=(19, 16, 12), spacing=0.45):
    """Nurse-call ding-dong: FM bells on a falling triad."""
    sec = spacing * len(notes) + 2.5
    out = np.zeros(n_(sec))
    for i, s in enumerate(notes):
        f = nt(s)
        d = 2.2
        t = np.arange(n_(d)) / SR
        idx = 2.5 * np.exp(-t / 0.3)
        x = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * 1.4 * t)) * perc(len(t), 0.001, 0.7)
        j = n_(i * spacing)
        out[j:j + len(x)] += x
    return out


def fetal_doppler(sec, bpm=144.0):
    """Doppler fetal monitor: a galloping whoosh, ~110-160 BPM, pitch wobbling with the flow."""
    t = np.arange(n_(sec)) / SR
    ph = (t * bpm / 60) % 1.0
    env = np.exp(-((ph - 0.1) / 0.06) ** 2) + 0.7 * np.exp(-((ph - 0.33) / 0.07) ** 2)
    centre = 380 + 380 * env
    x = tv_filter(noise(sec), centre, q=2.5, kind="bp")
    return x * (0.15 + env)


def mains_hum(f=60.0, sec=1.0, buzz=0.35):
    """The wall socket: 60 Hz + harmonics, 120 Hz transformer magnetostriction, a little saturation."""
    x = harmonics(f, sec, [1.0, 0.6, 0.35, 0.3, 0.16, 0.14, 0.08, 0.07])
    x = np.tanh((1 + buzz * 2) * x) / np.tanh(1 + buzz * 2)
    return x


def alarm_pad_tone(f, sec, detune_cents=7.0, attack=0.25, release=0.6):
    """Sustained chord voice made of the alarm-pulse harmonic recipe, three detuned copies."""
    out = np.zeros(n_(sec + release))
    for d in (-detune_cents, 0.0, detune_cents):
        x = harmonics(f * 2 ** (d / 1200), sec + release, [1.0, 0.5, 0.35, 0.22, 0.15], vibrato=0.0015,
                      vib_rate=4.0 + d / 10)
        out += x
    return out * ar(len(out), attack, release) / 3


# A catalogue for the sample pack (name -> zero-arg callable)
CATALOGUE = {
    "monitor_qrs_beep": lambda: qrs_beep(),
    "pulseox_desaturation_99_to_80": lambda: np.concatenate(
        [np.pad(pulseox_beep(s), (0, n_(0.25))) for s in range(99, 79, -1)]),
    "monitor_flatline": lambda: flatline(2.0),
    "iec_cardiac_high_priority": lambda: _burst("cardiac", "high"),
    "iec_ventilation_high_priority": lambda: _burst("ventilation", "high"),
    "iec_oxygen_high_priority": lambda: _burst("oxygen", "high"),
    "iec_drug_delivery_high_priority": lambda: _burst("drug", "high"),
    "iec_cardiac_medium_priority": lambda: _burst("cardiac", "medium"),
    "icon_heartbeat_lup_dup": lambda: np.concatenate([heartbeat(gap=0.22)] * 3),
    "icon_ventilation_breath": lambda: ventilator_breath(),
    "icon_temperature_kettle": lambda: kettle(3.0),
    "mri_gradient_knocks": lambda: np.concatenate([np.pad(mri_knock(), (0, n_(0.03))) for _ in range(12)]),
    "mri_epi_buzz": lambda: epi_buzz(1.5),
    "infusion_pump_clicks_and_motor": lambda: np.concatenate(
        [np.pad(pump_click(), (0, n_(0.2))) for _ in range(6)] + [pump_whir(1.0)]),
    "defib_charge_and_shock": lambda: np.concatenate([defib_charge(4.0), np.zeros(n_(0.3)), defib_shock()]),
    "humidifier_bubbles": lambda: np.concatenate([np.pad(bubble(f), (0, n_(0.09))) for f in
                                                  (420, 700, 520, 900, 610, 480, 800, 560)]),
    "oxygen_hiss": lambda: o2_hiss(1.5),
    "nurse_call_chime": lambda: nurse_chime(),
    "fetal_doppler": lambda: fetal_doppler(4.0),
    "mains_hum_60hz": lambda: mains_hum(60.0, 2.0) * trap(n_(2.0), 0.05, 0.2),
}


def _burst(category, priority):
    s16 = 60 / 128 / 4
    notes = iec_melody(category, priority)
    td = 0.11 if priority == "high" else 0.16
    out = np.zeros(n_(16 * s16 + 0.3))
    for on, semi in notes:
        x = iec_pulse(nt(semi), td)
        i = n_(on * s16)
        out[i:i + len(x)] += x
    return out


# ================================================================ 7 AM: the abrasive shift-change machines
def square(freq, sec):
    """Band-limited square (odd harmonics up to Nyquist): what a piezo driver chip outputs."""
    f = float(freq)
    ks = [k for k in range(1, 60, 2) if f * k < SR * 0.45]
    return harmonics(f, sec, [1.0 / k for k in ks], ratios=ks)


def piezo(f=3840.0, sec=0.12, drive=3.0):
    """Piezo buzzer (bed-exit, IV-pump occlusion): square drive at the disc's 2-4 kHz resonance,
    ~88-95 dB at 30 cm. 3840 Hz = 64 x 60 = C, 2880 Hz = G."""
    x = square(f, sec)
    x = x + reson(x, f, 6) * 1.5 + 0.25 * square(f / 2, sec)  # resonance ring + sub-rattle of the housing
    x = np.tanh(drive * x) / np.tanh(drive)
    return x * trap(len(x), 0.001, 0.004)


def occlusion_beep(f=1920.0, sec=0.18):
    """The fat, square 'BEEP' of an infusion pump occlusion or a portable X-ray exposure."""
    x = square(f, sec) + 0.6 * square(f / 2, sec)
    x = np.tanh(2.5 * x)
    return x * trap(len(x), 0.002, 0.01)


def nibp_inflate(sec=2.5, f=60.0):
    """Automatic blood-pressure cuff, inflating: a linear diaphragm pump stroking at mains frequency.
    It labours (pitch sags, grind rises) as cuff pressure climbs."""
    n = n_(sec)
    t = np.arange(n) / SR
    p = t / sec
    fr = f * (1 - 0.06 * p)
    ph = np.cumsum(fr) / SR % 1.0
    stroke = np.exp(-((ph - 0.15) / 0.07) ** 2)  # one hard stroke per cycle
    rattle = hp(noise(sec), 1800) * stroke * (0.3 + 0.7 * p)
    body = np.tanh(3 * (stroke - 0.3)) + 0.5 * np.sin(2 * np.pi * np.cumsum(fr) / SR)
    x = body * 0.8 + rattle * 0.6
    return x * trap(n, 0.02, 0.01) * (0.6 + 0.4 * p)


def nibp_step():
    """One deflation step: valve snap + a short bleed of air (monitors take ~10-12 of these)."""
    sec = 0.16
    t = np.arange(n_(sec)) / SR
    snap = reson(noise(sec) * perc(len(t), 0.0003, 0.002), 900, 5) * 2.5
    bleed = bp(noise(sec), 1500, 7000) * perc(len(t), 0.003, 0.05) * 0.5
    return snap + bleed


def pneumatic_tube(whoosh=1.4):
    """Pneumatic tube station: the carrier rushing up the pipe, then a slam into the station."""
    n = n_(whoosh)
    t = np.arange(n) / SR
    p = t / whoosh
    rush = tv_filter(noise(whoosh), 250 + 2500 * p ** 2, q=1.5, kind="bp") * p ** 2
    sec = 1.0
    tt_ = np.arange(n_(sec)) / SR
    slam = sine(38 + 60 * np.exp(-tt_ / 0.03), sec) * perc(len(tt_), 0.0005, 0.25)
    metal = (reson(noise(sec) * perc(len(tt_), 0.0005, 0.004), 420, 30)
             + reson(noise(sec) * perc(len(tt_), 0.0005, 0.004), 1130, 40) * 0.7) * 3
    crack = bp(noise(sec), 300, 9000) * perc(len(tt_), 0.0005, 0.02)
    hit = np.tanh(2 * (slam + metal * perc(len(tt_), 0.0005, 0.3) + crack))
    return np.concatenate([rush * 0.6, hit])


def floor_scrubber(sec):
    """Floor scrubber doing the corridor before rounds: brush rumble (~3 rev/s), vacuum roar, motor whine."""
    n = n_(sec)
    t = np.arange(n) / SR
    rumble = lp(noise(sec), 160) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)) * 3
    roar = bp(noise(sec), 300, 2500) * 0.35
    whine = harmonics(1150 * (1 + 0.004 * np.sin(2 * np.pi * 0.7 * t)), sec, [0.2, 0.08, 0.05])
    hum = harmonics(120.0, sec, [0.5, 0.3, 0.2])
    return rumble + roar + whine + hum


def xray_charge(sec):
    """Portable X-ray: the generator's rising whine before an exposure."""
    return defib_charge(sec, f_lo=300.0, f_hi=5200.0)


def hard_kick():
    """A distorted industrial kick with the heartbeat's 'lub' inside it."""
    sec = 0.5
    t = np.arange(n_(sec)) / SR
    f = 46 + 190 * np.exp(-t / 0.028)
    body = sine(f, sec) * perc(len(t), 0.0008, 0.2)
    click = hp(noise(sec), 2000) * perc(len(t), 0.0003, 0.004) * 0.5
    lub = _thump(50, sec, 0.06) * 0.5
    x = np.tanh(4.0 * (body + click + lub)) / np.tanh(4.0)
    return x * perc(len(t), 0.0005, 0.22) ** 0.5


def mains_saw(f, sec, n_harm=40):
    """Mains hum, but bright: every harmonic of the transformer buzz, not just the first few."""
    ks = [k for k in range(1, n_harm) if f * k < 9000]
    amps = [1.0 / k ** 0.9 * (1.4 if k % 2 == 0 else 1.0) for k in ks]
    return harmonics(f, sec, amps, ratios=ks)


def rolling_bass(f, sec, cutoff=900.0, drive=3.5):
    """One 16th of rolling bass: mains-hum saw, wave-shaped, plucked filter, plus a clean sub."""
    x = mains_saw(f, sec) + 0.5 * mains_saw(f * 1.006, sec) + 0.4 * mains_saw(f * 2, sec)
    x = np.tanh(drive * x * 0.6)
    n = len(x)
    e = perc(n, 0.002, sec * 0.45)
    x = lp(x * e, cutoff, order=4)
    x = np.tanh(2.0 * x)  # second stage of grit after the filter, keeps 150-800 Hz loud on small speakers
    sub = sine(f, sec) * perc(n, 0.003, sec * 0.7)
    return x * 0.9 + sub * 1.0


def reese(f, sec, cutoff=400.0):
    """Long, dark, detuned hum drone."""
    x = mains_saw(f, sec) + mains_saw(f * 1.011, sec) + mains_saw(f * 0.993, sec)
    x = np.tanh(1.5 * x / 3)
    return lp(x, cutoff, order=4) + 0.6 * sine(f, sec)


CATALOGUE.update({
    "7am_piezo_bed_exit": lambda: np.concatenate([np.pad(piezo(f, 0.12), (0, n_(0.04)))
                                                  for f in [3840, 2880] * 8]),
    "7am_infusion_occlusion_beep": lambda: np.concatenate([np.pad(occlusion_beep(), (0, n_(0.12)))] * 3),
    "7am_bp_cuff_cycle": lambda: np.concatenate([nibp_inflate(2.5)] + [np.pad(nibp_step(), (0, n_(0.35)))
                                                                         for _ in range(10)]),
    "7am_pneumatic_tube": lambda: pneumatic_tube(),
    "7am_floor_scrubber": lambda: floor_scrubber(4.0) * trap(n_(4.0), 0.5, 0.5),
    "7am_xray_charge_and_exposure": lambda: np.concatenate([xray_charge(3.0), occlusion_beep(1920, 0.6)]),
})
