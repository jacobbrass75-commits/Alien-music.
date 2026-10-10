"""Train a policy in one of the alien-music environments.

  python train.py rhythm
  python train.py composer --iters 2000 --set ear=human        # human-ear baseline for comparison
  python train.py signal --set n_attrs=3 noise=0.6 --ppo ent_coef=0.02
  python train.py composer --resume runs/composer/<run>/ckpt.pt --iters 1000
"""
import argparse
import json
import os
import time
from dataclasses import asdict, fields

import torch

from alien.envs import ENVS
from alien.ppo import Policy, PPOConfig, train
from alien.utils import get_device, parse_overrides, write_wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("env", choices=list(ENVS))
    p.add_argument("--iters", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0, help="seeds the timbre, ear, drums and the RNG")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", help="run folder name (default: timestamp)")
    p.add_argument("--set", nargs="*", default=[], help="environment kwargs, e.g. ear=human tension_quantile=0.6")
    p.add_argument("--ppo", nargs="*", default=[], help="PPO config, e.g. ent_coef=0.02 lr=1e-4")
    p.add_argument("--resume", help="checkpoint to continue from (uses its env settings)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    env_cls = ENVS[args.env]
    ckpt = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None

    env_kwargs = dict(ckpt["env_kwargs"]) if ckpt else {"seed": args.seed}
    env_kwargs.update(parse_overrides(args.set))
    if args.num_envs:
        env_kwargs["num_envs"] = args.num_envs
    cfg_kwargs = {**env_cls.ppo_defaults, **(ckpt["ppo"] if ckpt else {}), **parse_overrides(args.ppo)}
    if args.iters:
        cfg_kwargs["iters"] = args.iters
    cfg = PPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in {f.name for f in fields(PPOConfig)}})

    env = env_cls(device=device, **env_kwargs)
    policy = None
    if ckpt:
        env.load_state_dict(ckpt["env_state"])
        policy = Policy(env.obs_dim, env.action_dims, cfg.hidden).to(device)
        policy.load_state_dict(ckpt["policy"])

    out_dir = os.path.join(args.out, args.env, args.name or time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"env": args.env, "env_kwargs": env_kwargs, "ppo": asdict(cfg), "device": str(device),
                   "resumed_from": args.resume}, f, indent=2)
    with open(os.path.join(out_dir, "world.txt"), "w", encoding="utf-8") as f:
        f.write(env.describe())

    print(f"== {args.env} on {device} -> {out_dir}\n{env.describe()}\n")
    train(env, cfg, out_dir, env_kwargs, write_wav, policy)


if __name__ == "__main__":
    main()
