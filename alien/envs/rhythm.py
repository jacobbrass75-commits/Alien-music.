"""Rhythm environment.

Each step the agent emits one event: which of K alien percussion voices strike (2^K patterns,
0 = rest) and the inter-onset interval to the next event, picked from ratios spaced by sqrt(phi).
There is no beat grid, so no human meter is built in.

Reward (plus the surprise band from base.py):
  + compression progress of a predictor on the event sequence (structure without a template)
  - inverted-U density penalty (stops collapse to silence or to everything-all-the-time)
  - isochrony penalty: if one IOI dominates, it's a steady pulse, which is the most human thing
    a rhythm can do. Set isochrony_weight=0 to allow it.
"""
import numpy as np
import torch

from ..sound import Drum, phi_ratios, render_hits
from .base import TokenSeqEnv


class RhythmEnv(TokenSeqEnv):
    name = "rhythm"

    def __init__(self, num_envs=128, episode_len=64, window=16, n_voices=4, ioi_ratios=None, base_ioi=0.16,
                 density_target=0.3, density_width=0.15, density_weight=1.0, isochrony_weight=1.0,
                 isochrony_free=0.4, lp_weight=1.0, lp_kwargs=None, seed=0, device="cpu", **curiosity):
        self.ioi_ratios = list(ioi_ratios or phi_ratios(7))
        self.n_voices, self.base_ioi, self.seed = n_voices, base_ioi, seed
        self.density = (density_target, density_width, density_weight)
        self.isochrony = (isochrony_weight, isochrony_free)
        self.drums = [Drum.alien(seed, i) for i in range(n_voices)]
        super().__init__(num_envs, episode_len, window, [2 ** n_voices, len(self.ioi_ratios)], device,
                         lp_weight, lp_kwargs, **curiosity)
        self._bits = 2 ** torch.arange(n_voices, device=self.device)

    def extrinsic(self, tokens):
        B, T, _ = tokens.shape
        hits = ((tokens[..., 0:1] & self._bits) > 0).float()  # [B, T, K]
        target, width, w_d = self.density
        density = hits.mean((1, 2))
        r_density = -w_d * ((density - target) / width) ** 2

        # Fraction of sounding events that share the most common IOI.
        sounding = (tokens[..., 0] > 0).float()
        counts = torch.zeros(B, len(self.ioi_ratios), device=self.device)
        counts.scatter_add_(1, tokens[..., 1], sounding)
        iso = counts.max(1).values / sounding.sum(1).clamp(min=1)
        w_i, free = self.isochrony
        r_iso = -w_i * ((iso - free).clamp(min=0) / (1 - free)) ** 2 * 4

        r = (r_density + r_iso).unsqueeze(1).expand(B, T)  # per step, same scale as the curiosity terms
        distinct = np.mean([len(set(map(tuple, ep.tolist()))) for ep in tokens.cpu()])
        return r, {"density": density.mean().item(), "isochrony": iso.mean().item(), "distinct_events": distinct}

    def render(self, index, sr=22050):
        toks = self.last_tokens[index].cpu().tolist()
        hits, time = [], 0.0
        for pattern, ioi in toks:
            for k in range(self.n_voices):
                if pattern >> k & 1:
                    hits.append((time, k, 0.6))
            time += self.base_ioi * self.ioi_ratios[ioi]
        return render_hits(hits, self.drums, sr, seed=index), sr

    def describe(self):
        iois = ", ".join(f"{r:.3f}" for r in self.ioi_ratios)
        drums = "; ".join(f"v{i}: {d.f0:.0f}Hz x[{', '.join(f'{r:.2f}' for r in d.ratios)}]"
                          for i, d in enumerate(self.drums))
        return f"IOI ratios: {iois}\nDrums: {drums}"
