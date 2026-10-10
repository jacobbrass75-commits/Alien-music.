# MacBook handoff: alien-music

Picking up the alien-music project on the MacBook Pro while the PC is busy with Blender.

## Where things stand (Oct 10, 2026)

Three RL environments that train music with **no human music anywhere in the reward**. Rewards
come from a physical roughness model of a (non-human) ear and from curiosity (compression progress
plus a surprise band). See `README.md` for the full design.

First experiment ran on the PC (CPU, 16 min). The three trained models are in this repo under
`runs/<env>/first/ckpt.pt`:

| Model | Result | Status |
|---|---|---|
| **signal** (two agents evolve a sound language) | listener decodes **84%** of calls (chance 26%), energy halved | works, stable from ~iter 400 |
| **rhythm** (alien drums, no beat grid) | ~3–4 distinct events per piece, density 0.49 (target 0.3), still leans toward a steady pulse | alive but narrow |
| **composer** (scale derived from an alien timbre + ear) | best around **iter 400–600** (~10 chords/piece), narrowed to **2–3 chords** by iter 2000 | peaked early, then got repetitive |

Composer world, seed 0: "octave" = ratio 2.40 (1518 cents), scale = 0, 496, 621, 740, 879, 1116, 1235 cents.

## 1. Set up the Mac (once)

Needs an Apple Silicon Mac (M1 or newer); current PyTorch has no Intel-Mac builds.

```bash
git clone https://github.com/jacobbrass75-commits/Alien-music..git alien-music
cd alien-music
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

For the studio you also need Node.js (`brew install node` if missing).

## 2. Look at what we have (Xeno Studio)

```bash
python server.py                              # terminal 1 (venv active)
cd studio && npm install && npm run dev       # terminal 2, then open http://localhost:5173
```

Pick a model, Generate, play with the Sound lab. Signal models show their lexicon as a grid.

## 3. Next tasks, in order

### Task A: keep checkpoints along the way (code change)
Right now `alien/ppo.py` overwrites `ckpt.pt`, so the composer's best moment (iter ~500) was lost
as a model and only survives as audio in the sample WAVs. Change `save_checkpoint` calls in
`train()` to also write `ckpt_iter{it:05d}.pt` every N iterations (e.g. 250), and make
`server.py`'s `find_ckpts()` glob `ckpt*.pt` so the studio can load any of them.

### Task B: stop the composer from narrowing (tuning sweep)
Try stronger anti-repetition pressure. On the Mac use `--parallel 4` (6 on an M-series Pro/Max):

```bash
caffeinate -i python scripts/sweep.py composer --name tune1 --seeds 0 1 --iters 1500 --parallel 4 \
    --grid surprise_weight=1,2,4 ppo.ent_coef=0.01,0.03
python scripts/summarize.py runs/composer/tune1
```
Look for runs that keep `distinct_chords` above ~8 and `surprise` near 0.35 late in training
without entropy hitting 0. Listen to `runs/composer/tune1/<run>/samples/` before trusting numbers.

### Task C: richer signal-game language
Raise channel noise so fewer meanings fall back on silence:
```bash
caffeinate -i python train.py signal --iters 800 --name noisy --set noise=0.6
```

### Task D: rhythm
Same idea as the composer: try `surprise_weight=2` and a stronger `isochrony_weight`, and check
that density moves toward its 0.3 target.

## 4. Mac notes
- **Keep it awake and plugged in.** Prefix training with `caffeinate -i`.
- **CPU is right for these models.** `--device auto` uses CPU on a Mac; don't force `mps`
  (Apple's GPU backend), since per-step overhead likely makes it slower for models this small.
- `scripts/first_experiment.sh --wait` (idle detection) is Windows-only; on the Mac it just starts.
- Expect roughly 15–40 min for a three-model run, depending on the chip (estimate, not yet measured).

## 5. Getting results back to the PC
`runs/` is git-ignored because sample WAVs are bulky. To share a trained model, commit just its
small files:
```bash
git add -f runs/composer/tune1/<run>/{ckpt.pt,config.json,world.txt,log.jsonl}
git commit -m "Add composer tune1 model" && git push
```
Then `git pull` on the PC.

## Key files
```
alien/sound.py          ear (roughness), timbres, scale derivation, synths
alien/curiosity.py      predictor + compression progress
alien/ppo.py            PPO trainer (checkpointing lives here: Task A)
alien/envs/             base.py, rhythm.py, signal_game.py, composer.py
train.py, render.py     train one model / render WAVs from a checkpoint
scripts/sweep.py        grid x seeds in parallel; scripts/summarize.py compares runs
server.py + studio/     Xeno Studio
```
