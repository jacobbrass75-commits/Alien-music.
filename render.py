"""Generate pieces from a trained checkpoint.

  python render.py runs/composer/<run>/ckpt.pt --n 8
  python render.py runs/composer/<run>/ckpt.pt --n 8 --temperature 0.7   # more conservative
  python render.py runs/signal/<run>/ckpt.pt                              # hear the evolved lexicon
"""
import argparse
import os

import torch

from alien.envs import ENVS
from alien.ppo import Policy, rollout
from alien.utils import get_device, write_wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", help="defaults to <run>/renders")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    env_cls = ENVS[ckpt["env"]]
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "renders")

    if ckpt["env"] == "signal":
        probe = env_cls(device=device, **{**ckpt["env_kwargs"], "num_envs": 1})
        meanings = probe.all_meanings()
        env = env_cls(device=device, **{**ckpt["env_kwargs"], "num_envs": len(meanings)})
    else:
        env = env_cls(device=device, **{**ckpt["env_kwargs"], "num_envs": args.n})
    env.load_state_dict(ckpt["env_state"])
    env.learn = False
    policy = Policy(ckpt["obs_dim"], ckpt["action_dims"], ckpt["ppo"]["hidden"]).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    if ckpt["env"] == "signal":
        _, _, info = rollout(env, policy, args.temperature, meanings=meanings)
        audio, sr = env.render_lexicon(env.last_actions)
        write_wav(os.path.join(out, "lexicon.wav"), audio, sr)
        for m, a in zip(meanings.tolist(), env.last_actions.tolist()):
            print(f"meaning {m}: pitch/loud {a}")
        print(f"listener accuracy on the lexicon: {info['metrics']['listener_acc']:.2f}")
    else:
        _, _, info = rollout(env, policy, args.temperature)
        for i in range(args.n):
            audio, sr = env.render(i)
            write_wav(os.path.join(out, f"piece_{i}.wav"), audio, sr)
        print({k: round(v, 3) for k, v in info["metrics"].items()})
    print(f"wrote to {out}")


if __name__ == "__main__":
    main()
