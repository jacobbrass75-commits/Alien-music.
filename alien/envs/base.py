"""Vectorised, fixed-length token-sequence environment.

All B episodes run in lockstep for exactly T steps. Each step the agent emits one event made of
H categorical choices (a "token"). Rewards that need the whole piece (compression progress,
density, ...) are computed at the end and handed back as info["retro_reward"] of shape [T, B],
which the PPO trainer adds to the per-step rewards before computing advantages.

Every token env gets two curiosity terms for free:
  lp        compression progress: how much the predictor improved on this piece
  surprise  information rate, an inverted U (Berlyne's Wundt curve) on how surprised the predictor
            was by each event *before* learning from it, as a fraction of the maximum possible.
            Repetition (~0) and noise (~1) both lose. Without this term the policy collapses: early
            on, repetition is the easiest thing to make progress on, and once learned it pays zero
            but the policy has stopped exploring.
"""
import math

import torch
import torch.nn.functional as F

from ..curiosity import LearningProgress, onehot_tokens


def inverted_u(x, target, width):
    """Penalty <= 0, zero at target. Quadratic within one width, linear beyond (Huber), so the
    gradient never vanishes far from the target, which is where a fresh random policy starts."""
    z = (x - target).abs() / width
    return -torch.where(z < 1, 0.5 * z ** 2, z - 0.5)


class TokenSeqEnv:
    name = "base"
    ppo_defaults = {}

    def __init__(self, num_envs, episode_len, window, head_sizes, device, lp_weight=1.0, lp_kwargs=None,
                 surprise_target=0.35, surprise_width=0.15, surprise_weight=1.0):
        self.num_envs, self.episode_len, self.window = num_envs, episode_len, window
        self.head_sizes, self.device = list(head_sizes), torch.device(device)
        self.lp_weight = lp_weight
        self.surprise = (surprise_target, surprise_width, surprise_weight)
        self.max_surprisal = sum(math.log(n) for n in self.head_sizes)
        self.lp = LearningProgress(self.head_sizes, window, self.device, **(lp_kwargs or {}))
        self.tokens = torch.full((num_envs, episode_len, len(head_sizes)), -1, dtype=torch.long, device=self.device)
        self.last_tokens = None
        self.learn = True  # False = just generate: no predictor updates, only the cheap extrinsic metrics
        self.t = 0

    @property
    def obs_dim(self):
        return self.window * sum(self.head_sizes) + 1

    @property
    def action_dims(self):
        return self.head_sizes

    def _obs(self):
        sl = self.tokens[:, max(self.t - self.window, 0):self.t]
        oh = F.pad(onehot_tokens(sl, self.head_sizes), (0, 0, self.window - sl.shape[1], 0))
        phase = torch.full((self.num_envs, 1), self.t / self.episode_len, device=self.device)
        return torch.cat([oh.flatten(1), phase], -1)

    def reset(self):
        self.tokens.fill_(-1)
        self.t = 0
        return self._obs()

    def step(self, action):
        self.tokens[:, self.t] = action
        self.t += 1
        done = self.t == self.episode_len
        info = {}
        if done and not self.learn:
            info["metrics"] = self.extrinsic(self.tokens)[1]
            self.last_tokens = self.tokens.clone()
        elif done:
            lp, surprisal, metrics = self.lp(self.tokens)
            target, width, w_s = self.surprise
            frac = surprisal / self.max_surprisal
            r_surprise = w_s * inverted_u(frac, target, width)
            ext, ext_metrics = self.extrinsic(self.tokens)
            info["retro_reward"] = (self.lp_weight * lp + r_surprise + ext).T
            info["metrics"] = {**metrics, **ext_metrics, "lp": lp.mean().item(), "surprise": frac.mean().item()}
            self.last_tokens = self.tokens.clone()
        return self._obs(), torch.zeros(self.num_envs, device=self.device), done, info

    def extrinsic(self, tokens):
        """[B, T, H] -> ([B, T] reward, metrics dict). Override."""
        return torch.zeros(tokens.shape[:2], device=self.device), {}

    def render(self, index, sr=22050):
        raise NotImplementedError

    def describe(self):
        return ""

    def state_dict(self):
        return {"lp": self.lp.state_dict()}

    def load_state_dict(self, sd):
        self.lp.load_state_dict(sd["lp"])
