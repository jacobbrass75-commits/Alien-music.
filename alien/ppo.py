"""Minimal PPO for the lockstep, multi-discrete environments in alien.envs."""
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    iters: int = 500
    episodes_per_iter: int = 1
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 2048
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    sample_every: int = 25
    n_samples: int = 3
    save_every: int = 50


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.heads = nn.ModuleList(nn.Linear(hidden, n) for n in action_dims)
        self.value = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        for h in self.heads:
            nn.init.orthogonal_(h.weight, 0.01)
        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, obs):
        h = self.trunk(obs)
        return [head(h) for head in self.heads], self.value(h).squeeze(-1)


def sample(logits, temperature=1.0):
    dists = [Categorical(logits=l / temperature) for l in logits]
    a = torch.stack([d.sample() for d in dists], -1)
    return a, sum(d.log_prob(a[:, i]) for i, d in enumerate(dists))


def logp_entropy(logits, actions):
    dists = [Categorical(logits=l) for l in logits]
    return (sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists)),
            sum(d.entropy() for d in dists))


def gae(rewards, values, gamma, lam):
    adv, last = torch.zeros_like(rewards), 0.0
    for t in reversed(range(rewards.shape[0])):
        next_v = values[t + 1] if t + 1 < rewards.shape[0] else 0.0
        last = rewards[t] + gamma * next_v - values[t] + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def rollout(env, policy, temperature=1.0, **reset_kwargs):
    obs, buf = env.reset(**reset_kwargs), defaultdict(list)
    for _ in range(env.episode_len):
        with torch.no_grad():
            logits, v = policy(obs)
            a, lp = sample(logits, temperature)
        buf["obs"].append(obs); buf["act"].append(a); buf["logp"].append(lp); buf["val"].append(v)
        obs, r, done, info = env.step(a)
        buf["rew"].append(r)
    rewards = torch.stack(buf["rew"])
    if "retro_reward" in info:
        rewards = rewards + info["retro_reward"]
    return {k: torch.stack(v) for k, v in buf.items() if k != "rew"}, rewards, info


def save_checkpoint(path, env, policy, env_kwargs, cfg, it):
    torch.save({"env": env.name, "env_kwargs": env_kwargs, "ppo": asdict(cfg), "iter": it,
                "obs_dim": env.obs_dim, "action_dims": env.action_dims,
                "policy": policy.state_dict(), "env_state": env.state_dict()}, path)


def train(env, cfg, out_dir, env_kwargs, write_wav, policy=None):
    device = env.device
    policy = policy or Policy(env.obs_dim, env.action_dims, cfg.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    reward_std = None
    it = 0
    try:
        for it in range(1, cfg.iters + 1):
            t0 = time.time()
            batches, metrics, raw_returns = [], defaultdict(list), []
            for _ in range(cfg.episodes_per_iter):
                buf, rewards, info = rollout(env, policy)
                raw_returns.append(rewards.sum(0).mean().item())
                for k, v in info.get("metrics", {}).items():
                    metrics[k].append(v)
                s = rewards.std().item()
                reward_std = s if reward_std is None else 0.99 * reward_std + 0.01 * s
                adv, ret = gae(rewards / (reward_std + 1e-8), buf["val"], cfg.gamma, cfg.lam)
                batches.append({"obs": buf["obs"], "act": buf["act"], "logp": buf["logp"], "adv": adv, "ret": ret})

            data = {k: torch.cat([b[k].flatten(0, 1) for b in batches]) for k in batches[0]}
            data["adv"] = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
            n = data["obs"].shape[0]
            stats = defaultdict(list)
            for _ in range(cfg.epochs):
                for idx in torch.randperm(n, device=device).split(cfg.minibatch):
                    logits, v = policy(data["obs"][idx])
                    logp, ent = logp_entropy(logits, data["act"][idx])
                    ratio = (logp - data["logp"][idx]).exp()
                    adv = data["adv"][idx]
                    pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                    vf = (v - data["ret"][idx]).pow(2).mean()
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent.mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    opt.step()
                    stats["entropy"].append(ent.mean().item())
                    stats["kl"].append((data["logp"][idx] - logp).mean().item())
                    stats["vf_loss"].append(vf.item())

            row = {"iter": it, "return": float(np.mean(raw_returns)), "sec": round(time.time() - t0, 2),
                   **{k: float(np.mean(v)) for k, v in stats.items()},
                   **{k: float(np.mean(v)) for k, v in metrics.items()}}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(" | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

            if it % cfg.sample_every == 0 or it == cfg.iters:
                for i in range(min(cfg.n_samples, env.num_envs)):
                    audio, sr = env.render(i)
                    write_wav(os.path.join(out_dir, "samples", f"iter{it:05d}_{i}.wav"), audio, sr)
            if it % cfg.save_every == 0:
                save_checkpoint(os.path.join(out_dir, "ckpt.pt"), env, policy, env_kwargs, cfg, it)
    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint.")
    save_checkpoint(os.path.join(out_dir, "ckpt.pt"), env, policy, env_kwargs, cfg, it)
    log.close()
    return policy
