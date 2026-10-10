# alien-music

RL environments for music with no human reference point. There's no dataset and no pretraining.
Every reward comes from physics (a roughness model of an ear) or from information theory
(curiosity), never from "sounds like human music".

## The three environments

| env | what the agent does | what it's rewarded for |
|---|---|---|
| `rhythm` | each step: which of 4 alien drums strike + the gap to the next event, from sqrt(phi)-spaced ratios (no beat grid) | compression progress, a mid-band of surprise, a target density, and *not* settling into a steady pulse |
| `signal` *(the fun one)* | a speaker makes an 8-frame call to tell a listener a hidden meaning through a noisy alien ear | listener decodes it correctly, minus an energy cost. The two agents invent a sound language from nothing. |
| `composer` *(the goal one)* | each step: a pitch-or-rest per voice + a duration. Pitches come from a scale **derived** from a stretched-partial timbre's dissonance curve as heard by a non-human ear, so the "octave" isn't 2:1 | compression progress, a mid-band of surprise, and a target level of alien-ear roughness (tension, not consonance) |

`composer` is the one aimed squarely at the "Move 37" goal. The timbre, ear, scale and period all
fall out of `--seed`. Each seed is a different alien world: look at `world.txt` in the run folder.

### Why the rewards look like this
- **Compression progress** (`alien/curiosity.py`): a predictor learns to predict the next event.
  Reward = how much it improved on your piece. Noise and repetition both pay ~0.
- **Surprise band** (`alien/envs/base.py`): an inverted U on how surprised the predictor was
  before learning. Needed in practice: with compression progress alone, the policy collapsed to one
  repeated event within ~100 iterations (repetition is the easiest thing to make early progress on).
- Every extrinsic term is an **inverted U, never a maximum**. Anything with a maximum gets hacked.

## Setup

CPU works for everything (all three envs were tested on CPU). The GPU pays off when you raise `--num-envs` into the
thousands or make the predictor/policy bigger.

```bash
pip install -r requirements.txt
```

### RX 9070 XT (when the GPU is free)
Use Linux or WSL2 (Ubuntu). Native Windows ROCm PyTorch is a preview and slow.
1. Install the AMD Radeon driver for WSL / ROCm per AMD's docs (the 9070 XT is on the supported list).
2. Install the ROCm build of PyTorch: pick *Linux / Pip / ROCm* on https://pytorch.org/get-started/locally/
   and run the command it gives you (`pip install torch --index-url https://download.pytorch.org/whl/rocm<ver>`).
3. Check it: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   (ROCm shows up through the `torch.cuda` API, which is expected).
4. Train as usual. `--device auto` picks the GPU. Scale up, e.g. `--num-envs 2048 --ppo minibatch=16384 hidden=512`.

### RunPod
Rent an **RTX 4090 on Community Cloud** (~$0.34/hr, mid 2026). The models here are tiny, so an A100 or H100 is
wasted money until you move to audio-codec-scale models. The way to use the card fully is many runs at once,
not one big run.

1. Create a pod from an official **PyTorch** template, with a network volume mounted at `/workspace`
   (the only storage that survives the pod stopping).
2. Get the code onto it, either by `git clone` of your (private) repo into `/workspace`, or with RunPod's
   file transfer: `runpodctl send alien-music.zip` locally, then `runpodctl receive <code>` on the pod.
3. `bash scripts/runpod_setup.sh` checks the GPU and runs a smoke test.
4. Run a sweep inside `tmux` so it survives a dropped SSH session:
   ```bash
   python scripts/sweep.py composer --name tune1 --seeds 0 1 2 --iters 2000 --parallel 12 \
       --grid surprise_weight=1,2 ppo.ent_coef=0.01,0.02
   python scripts/summarize.py runs/composer/tune1
   ```
5. Bring the results home (`runpodctl send runs/composer/tune1`), then **stop the pod**. Billing runs
   until you do, and checkpoints let you resume later with `--resume`.

Watch GPU use with `nvidia-smi`. If utilisation is low, raise `--parallel` or `--num-envs`.

## Usage

```bash
python train.py composer                       # 500 iterations, samples every 25 in runs/composer/<time>/samples
python train.py rhythm --iters 2000
python train.py signal --set n_attrs=3 noise=0.6
python train.py composer --seed 7              # a different alien world
python train.py composer --set ear=human timbre=harmonic   # human baseline, for an A/B listen
python train.py composer --resume runs/composer/<run>/ckpt.pt --iters 1000

python render.py runs/composer/<run>/ckpt.pt --n 8 --temperature 0.8
python render.py runs/signal/<run>/ckpt.pt     # lexicon.wav: one call per meaning, in order
```

`--set` takes any env constructor kwarg (see each file in `alien/envs/`); `--ppo` takes any
`PPOConfig` field (`alien/ppo.py`). Ctrl-C saves a checkpoint.

Each run writes `config.json`, `world.txt` (the derived timbre, ear and scale), `log.jsonl`
(one row per iteration), `samples/*.wav`, and `ckpt.pt`.

## Xeno Studio (play with the models)
A browser tool for using trained models. The models only output events; the studio synthesises
them with each model's own timbre and drums, so you can reshape the sound live without retraining.

```bash
python server.py                 # terminal 1: model backend on :8765 (finds every runs/**/ckpt.pt)
cd studio && npm run dev         # terminal 2: open http://localhost:5173
```
(Or `cd studio && npm run build` once, after which `python server.py` alone serves it at :8765.)

- **Generate**: pick a checkpoint, number of pieces, length (can go past the trained length),
  temperature (lower = safer, higher = wilder), seed (click the seed tag to reuse one).
- **Sound lab**: tempo, transpose, partial stretch (bends the timbre toward or away from its
  derived scale's consonances), brightness, envelope, drum tune/ring/noise, reverb, width.
  Changes re-trigger whatever is playing. **mutate** randomly nudges everything.
- **World**: the derived scale drawn against the human 12-note grid, the timbre's partials
  against the harmonic series, and the drums' modes.
- **Signal models** show their lexicon as a grid. Click a call to hear it, shift-click to
  build a sentence, and **babble** plays two speakers chattering.
- **Mix** layers pieces from any models (e.g. a rhythm under a composer piece); loop it or export a WAV.
- **Training** panel shows live runs; new checkpoints appear in the model list as they save.

## Reading the logs
- `entropy` near 0 means the policy collapsed. Raise `--ppo ent_coef=0.02` or `--set surprise_weight=2`.
- `surprise` is the fraction of max surprisal, and the band targets 0.35. Stuck near 1 means noise; near 0 means a loop.
- `lp` is compression progress, normalised. Bursts mean the agent found something new that's learnable.
- `distinct_events` / `distinct_chords`: vocabulary size per piece.
- signal: `listener_acc` (chance = 1/n_values), `energy` (should fall as the code gets efficient).

## Human in the loop, at the end only
Train with no human taste anywhere, then *curate*: render many pieces, keep the ones that do
something to you. Don't feed preferences back into the reward, or it drifts back to the middle of human taste.

## Layout
```
alien/sound.py         ear (roughness), timbres, scale derivation, synths
alien/curiosity.py     predictor + compression progress
alien/ppo.py           PPO for lockstep multi-discrete envs
alien/envs/            base.py, rhythm.py, signal_game.py, composer.py
train.py, render.py
scripts/sweep.py       grid x seeds in parallel (the way to use a rented GPU)
scripts/summarize.py   compare runs, flags collapse / off-band surprise
scripts/runpod_setup.sh
scripts/first_experiment.sh   the first three models (--wait holds until the PC is idle)
server.py              studio backend: lists checkpoints, generates events
studio/                Xeno Studio (Vite): synthesis, sound lab, lexicon, mix
```
