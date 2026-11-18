# Sinus Rhythm: hospital machines sing in a resonant language

An EDM track (128 BPM, 2:41) built only from **synthesised** hospital machines. There are no
recordings: every sound is generated from its published spec. The machines don't just play the
track, they **speak** it, in a source-filter language called **Telemetric**.

```bash
python -m hospital.song             # -> hospital/out/sinus_rhythm.wav + .mp3   (~40 s on a laptop CPU)
python -m hospital.song --samples   # also the sample pack: one WAV per machine + one per Telemetric word
```
Needs `numpy` and `scipy` only (no torch). Listen: [`renders/sinus_rhythm.mp3`](renders/sinus_rhythm.mp3).
Every machine on its own: [`renders/samples/`](renders/samples/).

## 1. The research: what hospital machines actually sound like

| machine | what the research says | how it's synthesised here (`machines.py`) |
|---|---|---|
| **IEC 60601-1-8 alarm pulses** | Each pulse has a fundamental between **150 and 1000 Hz** plus **at least 4 harmonics** in 300–4000 Hz. **High priority** is 10 fast pulses in a **3 + 2, 3 + 2** pattern. **Medium** is 3 slower pulses, and **low** is 1–2. | `iec_pulse`: 5 harmonics, trapezoid envelope with 15 % rise/fall. On a 16th-note grid at 128 BPM, the high-priority burst `x x x . x x . . x x x . x x . .` fills **exactly one bar**, and the medium-priority burst is a **tresillo**. |
| **IEC Annex F melodies** (2006 edition) | Each alarm category has its own melody. Cardiac is `c e g` (medium), escalating to `c e g – g C` (high). | `IEC_MELODIES`: cardiac, ventilation, oxygen, drug delivery, temperature, perfusion, power and general. These melodies are the song's hooks. |
| **IEC 60601-1-8:2020 auditory icons** | The 2020 amendment replaces many tones with sounds you can recognise. **Cardiovascular** is a "lup-dup" heartbeat. **Ventilation** is a ~1 s white-noise inhale, a 0.5 s gap, then a slow exhale. **Temperature** is a whistling kettle. Icons sit in 200–5000 Hz. | `heartbeat` (it's inside the kick, and its "dup" lands one 16th later), `ventilator_breath`, `kettle` (the breakdown riser). |
| **ICU ventilator** | The default ICU **I:E ratio is 1:2**, and set rates run ~3–30 breaths/min. | One breath = 2 bars = 3.75 s, which is **16 breaths/min**. The pads breathe with it. |
| **Pulse oximeter** | The beep's pitch tracks SpO2, but the mapping isn't standardised. One patent uses a log mapping (662 Hz at 100 %, **halving every 24 points**). A survey of 21 oximeters found 479–921 Hz at 100 %. | `spo2_freq`: the log mapping, so **1 % SpO2 = one quarter-tone**. In the breakdown, saturation falls from 99 to 76 %, so the beep drops an octave in quarter-tone steps. |
| **ECG monitor QRS beep** | A short tone on every detected heartbeat. Its pitch isn't fixed, and some monitors pitch it by SpO2. | `qrs_beep`: the beeps sit at 64 BPM in the intro, rise to 128 in the build, and reach tachycardia 16ths before the flatline. |
| **Flatline (asystole)** | A continuous tone. | `flatline`: 960 Hz (16 × 60). |
| **MRI scanner** | In one 3 T study, the gradient coils resonate near **730 Hz**, with wider bands at **3–4 kHz** and **5.5–6.5 kHz**. The EPI readout tone measured **~1.9 kHz**, with harmonics at **3.8, 5.8, 7.7 and 9.6 kHz**. Levels reach 90–130 dB. | `mri_knock` (an impulse into those modes): percussion and rolls. `epi_buzz` (1920 Hz + harmonics): scanner stabs. |
| **Defibrillator** | The charging whine comes from the inverter and transformer, not the capacitor, and it rises in pitch as the capacitor charges. No source gave an exact frequency. | `defib_charge`: pitch follows capacitor voltage (1 − e^(−t/τ)). It's the riser into drop 2. `defib_shock` is the drop impact and `zap` is the snare. |
| **Fetal doppler** | ~110–160 BPM. A galloping whoosh, because the device hears a Doppler shift, not the beat itself. | `fetal_doppler`: band-passed noise whose centre moves with each beat, at 144 BPM (a 9:8 polyrhythm against the track). |
| **Infusion pump, humidifier, O2 flowmeter, nurse call** | General descriptions only (stepper ticks, bubbling, hiss, ding-dong chimes). Not standardised. | `pump_click` (hi-hats), `bubble` (rising-pitch pops), `o2_hiss` (open hats and crashes), `nurse_chime` (FM bells). |
| **Mains hum** | 60 Hz (US) plus harmonics, and 120 Hz from transformer magnetostriction. | `mains_hum`: **the bass**. |

### The tuning falls out of the research
The track is in **C = 240 Hz = 4 × 60 Hz**, so the wall-socket hum *is* the bass note C. Two of
the measured machine frequencies then land in key:
- MRI gradient resonance **~730 Hz** ≈ **G** (720 Hz = 12 × 60)
- MRI EPI readout tone **~1.9 kHz** ≈ **C** (1920 Hz = 32 × 60)

## 2. Telemetric, the resonant language

Speech is a source passing through resonant filters, so the machines get a voice the same way (`language.py`):

| part of a voice | in Telemetric |
|---|---|
| throat (source) | a buzzing harmonic series, like the EPI readout tone (voiced), or ventilator airflow (whispered) |
| melody | the IEC alarm melodies, note for note |
| mouth (filter) | four narrow resonators (formants F1–F4) that glide between vowels |
| consonants | the machines' own transients |

**Grammar:** every alarm melody is a word, and every pulse is a syllable. Each syllable is built like this:
- **consonant = which machine is talking** (the alarm category)
- **vowel = the pulse's pitch** (the scale degree)

| consonant | machine | | pitch | vowel |
|---|---|---|---|---|
| k | MRI knock (general) | | c | u |
| d | heartbeat (cardiac) | | d | o |
| h | ventilator breath (ventilation) | | e | a |
| s | O2 hiss (oxygen) | | f, f# | ae |
| f | kettle steam (temperature) | | g | e |
| t | pump click (drug delivery) | | a, b | i |
| z | defib buzz (power) | | C | u |
| w | pump whir (perfusion) | | | |

The lexicon (`.` is a rest on the 16th grid):

| word | medium priority | high priority (one bar) |
|---|---|---|
| cardiac | `du . . da . . de` | `du da de . de du . . du da de . de du` |
| ventilation | `hu . . hi . . hae` | `hu hi hae . hi hae . . (repeat)` |
| oxygen | `su . . si . . si` | `su si si . se sae . . (repeat)` |
| drug delivery | `tu . . to . . te` | `tu to te . tu to . . (repeat)` |
| temperature | `fu . . fo . . fa` | `fu fo fa . fae fe . . (repeat)` |
| perfusion | `wu . . wae . . wu` | `wu wae wu . wu wae . . (repeat)` |
| power | `zu . . zu . . zu` | `zu zu zu . zu zu . . (repeat)` |
| general | `ku . . ku . . ku` | `ku ku ku . ku ku . . (repeat)` |

In drop 2 the machines hold a conversation. Each word sits on the chord its own melody spells:

| chord | Am | F | C | G |
|---|---|---|---|---|
| word | cardiac `du da de` (c e g ⊂ Am7) | ventilation `hu hi hae` (c a f = F) | drug `tu to te` (c d g = Csus2) | oxygen `su si si se sae` (C b a g f ⊂ G9) |

## 3. The song

| bars | time | section | what happens |
|---|---|---|---|
| 1–8 | 0:00 | **Intro: the ward at night** | 60 Hz hum, ventilator breathing, QRS beep at a resting 64 BPM, fetal doppler, nurse chime. The ventilator whispers `hu hi hae`. |
| 9–16 | 0:15 | **Build: heart rate rising** | The QRS beep doubles to 128 BPM, the heartbeat becomes the kick, and the pump clicks speed up. Cardiac medium-priority `du . . da . . de` in tresillo. Then an MRI knock roll, the oxygen-hiss riser, and the raw high-priority alarm as the call. |
| 17–32 | 0:30 | **Drop 1: code** | Kick with a heartbeat inside, defib-zap snare, pump-click hats and O2-hiss open hats. MRI knock percussion and the mains-hum bass, sidechained. The talking lead is the cardiac high-priority alarm spoken every bar. EPI stabs come in for the second half. |
| 33–44 | 1:00 | **Breakdown: desaturation** | The four words sung slowly. The pulse-ox falls 99 → 76 %, an octave down in quarter-tones. The oxygen alarm `su si si` joins below 90 %, then the kettle boils, the heart races and the scanner rolls. |
| 45–48 | 1:22 | **Flatline** | Everything stops but the 960 Hz flatline. The defib charges, the power word `zu zu zu` rides the whine, there's a "charged" tone, then a beat of silence. |
| 49–72 | 1:30 | **Drop 2: the machines talk** | The shock lands on the downbeat. The four-word conversation runs over Am F C G, doubled an octave down and with dry alarm pulses on top for the last 8 bars. |
| 73–84 | 2:15 | **Outro: sinus rhythm** | The drums leave and the heart settles back to 64 BPM. The ventilator whispers again, a last nurse chime plays, and the hum fades. |

## Sources
- IEC 60601-1-8 pulse and harmonic requirements, priority patterns, cardiac melody:
  [Digi-Key: implementing audible alarms](https://www.digikey.com/en/articles/how-to-properly-implement-audible-alarms-in-medical-monitoring),
  [TI SSZT261](https://edgeworker.ti.com/lit/pdf/sszt261),
  [standards.har-el.com, 6.3.3.1](https://standards.har-el.com/Projects/181701/60601-1-8/html/A2SubCl-6-3-3-1-a-priority-encoded.htm),
  [MDDI: audible alarms](https://www.mddionline.com/components/audible-alarms-in-medical-equipment)
- 2020 auditory icons (heartbeat, ventilation, kettle):
  [AAMI alarm sounds review](https://www.aami.org/misc/alarm-sounds-review),
  [PUI Audio: updates to IEC 60601-1-8](https://puiaudio.com/news/key-updates-to-iec-60601-1-8-what-engineers-need-to-know),
  [Same Sky guide](https://www.sameskydevices.com/blog/a-guide-to-iec-60601-1-8-and-medical-alarm-systems)
- Pulse-oximeter pitch:
  [US10687763B2](https://patents.google.com/patent/US10687763),
  [The sounds of desaturation (survey of 21 oximeters)](https://experts.arizona.edu/en/publications/the-sounds-of-desaturation-a-survey-of-commercial-pulse-oximeter-/),
  [ICAD 2019 paper 70](https://icad2019.icad.org/wp-content/uploads/2019/06/ICAD_2019_paper_70.pdf)
- MRI acoustic noise:
  [Frontiers in Physics 2022](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.907619/full),
  [PMC6872037](https://pmc.ncbi.nlm.nih.gov/articles/PMC6872037),
  [ISMRM 2009 #3029](https://cds.ismrm.org/protected/09MProceedings/PDFfiles/03029.pdf)
- Ventilator I:E: [Critical Care 2023, 27:4667](https://ccforum.biomedcentral.com/articles/10.1186/s13054-023-04667-4);
  breath-rate range: [US5836302](https://uspto.report/patent/grant/5836302)
- Defibrillator whine: [Physics Forums](https://www.physicsforums.com/threads/what-causes-the-charging-sound-of-a-capacitor.224679)
- Fetal doppler: [Johns Hopkins OB/GYN 101](https://oacapps.med.jhmi.edu/OBGYN-101/Text/Pregnancy/fetal_heart_beat.htm)

**Caveats.** The search results confirmed only the cardiac melody. The other seven Annex F melodies
come from the 2006 edition's table as I know it, because the pages that print the full table
couldn't be fetched from the build environment. Check them against the standard before treating
them as reference. The 2020 icon descriptions come from a draft (CD3) review. The pulse-ox
mapping is one vendor's design. Nothing here is a recording of a real device, so none of these
sounds should be used as real alarms.
