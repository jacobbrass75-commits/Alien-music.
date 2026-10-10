"""Backend for Xeno Studio (studio/). Lists trained checkpoints and has them generate pieces.

  python server.py                       # http://127.0.0.1:8765, serves studio/dist if built
  python server.py --runs runs other_runs

The models only output events (pitches, hits, durations) plus a description of their world
(timbre, ear, scale, drums). Sound is synthesised in the browser so it can be reshaped live.
"""
import argparse
import glob
import json
import os
import threading
import time
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import torch

from alien.envs import ENVS
from alien.ppo import Policy, rollout
from alien.sound import cents

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "studio", "dist")
LOCK = threading.Lock()  # torch modules and cached envs are shared between requests
_ckpts, _envs = {}, {}
RUN_DIRS = []


def load_ckpt(path):
    mtime = os.path.getmtime(path)
    if path not in _ckpts or _ckpts[path][0] != mtime:
        _ckpts[path] = (mtime, torch.load(path, map_location="cpu", weights_only=False))
    return _ckpts[path]


def find_ckpts():
    found = []
    for root in RUN_DIRS:
        found += glob.glob(os.path.join(root, "**", "ckpt.pt"), recursive=True)
    return sorted(set(os.path.abspath(p) for p in found))


def list_models():
    out = []
    for path in find_ckpts():
        try:
            mtime, ck = load_ckpt(path)
        except Exception as e:  # a checkpoint being written right now, or a stale format
            print(f"skip {path}: {e}")
            continue
        run_dir = os.path.dirname(path)
        out.append({"id": path, "env": ck["env"], "iter": ck["iter"], "seed": ck["env_kwargs"].get("seed", 0),
                    "run": os.path.relpath(run_dir, ROOT).replace("\\", "/"), "updated": mtime,
                    "episode_len": ck["env_kwargs"].get("episode_len")})
    return out


def runs_status():
    """Progress of every run that has a log, for the studio's training panel."""
    out = []
    for root in RUN_DIRS:
        for log in glob.glob(os.path.join(root, "**", "log.jsonl"), recursive=True):
            run_dir = os.path.dirname(log)
            try:
                cfg = json.load(open(os.path.join(run_dir, "config.json")))
                with open(log, "rb") as f:
                    lines = f.read().decode().strip().splitlines()
                last = json.loads(lines[-1]) if lines else {}
            except (OSError, ValueError):
                continue
            out.append({"run": os.path.relpath(run_dir, ROOT).replace("\\", "/"), "env": cfg["env"],
                        "iter": last.get("iter", 0), "iters": cfg["ppo"]["iters"],
                        "updated": os.path.getmtime(log), "last": last})
    return sorted(out, key=lambda r: -r["updated"])


def get_env(path, n, length):
    mtime, ck = load_ckpt(path)
    key = (path, mtime, n, length)
    if key not in _envs:
        cls, kw = ENVS[ck["env"]], dict(ck["env_kwargs"])
        if ck["env"] == "signal":
            kw["num_envs"] = 1
            kw["num_envs"] = len(cls(device="cpu", **kw).all_meanings())
        else:
            kw["num_envs"] = n
            if length:
                kw["episode_len"] = length
        env = cls(device="cpu", **kw)
        env.load_state_dict(ck["env_state"])
        env.learn = False
        policy = Policy(ck["obs_dim"], ck["action_dims"], ck["ppo"]["hidden"])
        policy.load_state_dict(ck["policy"])
        policy.eval()
        _envs.clear()  # keep one live env; they're cheap to rebuild
        _envs[key] = (env, policy)
    return _envs[key]


def world(env):
    w = {"describe": env.describe()}
    if env.name == "rhythm":
        w.update(ioi_ratios=env.ioi_ratios, base_ioi=env.base_ioi, n_voices=env.n_voices,
                 drums=[asdict(d) for d in env.drums])
    else:
        w.update(pitches=env.pitches, scale_cents=[cents(s) for s in env.scale], period=env.period,
                 period_cents=cents(env.period), timbre=asdict(env.timbre), ear=asdict(env.ear))
    if env.name == "composer":
        w.update(n_voices=env.n_voices, dur_ratios=env.dur_ratios, base_dur=env.base_dur)
    if env.name == "signal":
        w.update(frame_s=env.frame_s, n_loud=env.n_loud, n_attrs=env.n_attrs, n_values=env.n_values,
                 msg_len=env.msg_len)
    return w


def generate(body):
    path = body["model"]
    if path not in find_ckpts():  # only ever unpickle checkpoints we found ourselves
        raise ValueError("unknown model")
    n = max(1, min(int(body.get("n", 6)), 16))
    length = body.get("length")
    length = max(8, min(int(length), 512)) if length else None
    temperature = max(0.05, float(body.get("temperature", 1.0)))
    seed = int(body.get("seed", time.time_ns() % 2 ** 31))
    with LOCK:
        env, policy = get_env(path, n, length)
        torch.manual_seed(seed)
        if env.name == "signal":
            meanings = env.all_meanings()
            _, _, info = rollout(env, policy, temperature, meanings=meanings)
            pieces = [{"meaning": m, "frames": a} for m, a in zip(meanings.tolist(), env.last_actions.tolist())]
        else:
            _, _, info = rollout(env, policy, temperature)
            pieces = env.last_tokens.tolist()
        return {"env": env.name, "seed": seed, "temperature": temperature, "world": world(env),
                "pieces": pieces, "metrics": info.get("metrics", {})}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC, **kwargs)

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/models"):
            with LOCK:
                return self._json(list_models())
        if self.path.startswith("/api/status"):
            return self._json(runs_status())
        if self.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        if not os.path.isdir(STATIC):
            return self._json({"error": "studio not built: run `npm run dev` in studio/ or `npm run build`"}, 404)
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/generate"):
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            return self._json(generate(body))
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def log_message(self, fmt, *args):
        if "/api/generate" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--runs", nargs="+", default=[os.path.join(ROOT, "runs")])
    args = p.parse_args()
    RUN_DIRS.extend(os.path.abspath(r) for r in args.runs)
    torch.set_num_threads(2)  # generation is light; leave the cores to training
    print(f"Xeno Studio backend on http://127.0.0.1:{args.port}  (runs: {', '.join(RUN_DIRS)})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
