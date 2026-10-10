"""Alien composer: the environment aimed most directly at 'music with no human reference point'.

Setup, all derived from physics, none from human music:
  timbre  stretched-partial spectrum (seeded), so no octave exists
  ear     roughness model with non-human critical bands
  scale   the most prominent minima of that timbre's dissonance curve *as heard by that ear*;
          the deepest minimum becomes the period, the alien 'octave'
Each step the agent writes one sonority: a pitch (or rest) per voice plus a duration from
sqrt(phi)-spaced ratios. Holding the same pitch in a voice sustains the note.

Reward (plus the surprise band from base.py):
  + compression progress on the event sequence (structure, never defined)
  - inverted-U on alien-ear roughness: aim for a tension level, not for zero or maximum
"""
import itertools

import numpy as np
import torch

from ..sound import cents, derive_scale, make_ear, make_timbre, phi_ratios, render_notes
from .base import TokenSeqEnv, inverted_u


class ComposerEnv(TokenSeqEnv):
    name = "composer"

    def __init__(self, num_envs=128, episode_len=48, window=12, n_voices=3, n_registers=3, scale_size=7,
                 dur_ratios=None, base_dur=0.28, f_base=110.0, ear="alien", timbre="alien",
                 tension_quantile=0.35, tension_weight=1.0, lp_weight=1.0, lp_kwargs=None, seed=0, device="cpu", **curiosity):
        self.timbre, self.ear = make_timbre(timbre, seed), make_ear(ear, seed)
        self.scale, self.period = derive_scale(self.timbre, self.ear, f0=f_base * 2, max_notes=scale_size)
        self.pitches = [f_base * self.period ** k * s for k in range(n_registers) for s in self.scale]
        self.dur_ratios = list(dur_ratios or phi_ratios(5))
        self.n_voices, self.base_dur, self.tension_weight = n_voices, base_dur, tension_weight
        M = len(self.pitches)
        super().__init__(num_envs, episode_len, window, [M + 1] * n_voices + [len(self.dur_ratios)], device,
                         lp_weight, lp_kwargs, **curiosity)
        self._build_tension_table(tension_quantile)

    def _build_tension_table(self, q):
        """Roughness of every possible sonority, looked up by flat index during training."""
        M, V = len(self.pitches), self.n_voices
        chords = torch.tensor(list(itertools.product(range(M + 1), repeat=V)))  # [C, V], 0 = rest
        hz = torch.tensor([1.0] + self.pitches, dtype=torch.float64)
        ratios = torch.tensor(self.timbre.ratios, dtype=torch.float64)
        amps = torch.tensor(self.timbre.amps, dtype=torch.float64)
        table = []
        for chunk in chords.split(20000):
            f = (hz[chunk][..., None] * ratios).flatten(1)
            a = ((chunk > 0).double()[..., None] * amps).flatten(1)
            table.append(self.ear.roughness(f, a))
        table = torch.cat(table)
        sounding = table[(chords > 0).sum(1) >= 2]
        self.tension_target = torch.quantile(sounding, q).item()
        self.tension_width = sounding.std().item()
        self.tension_table = (table / self.tension_width).float().to(self.device)
        self.tension_target /= self.tension_width
        self._place = (M + 1) ** torch.arange(V - 1, -1, -1, device=self.device)

    def extrinsic(self, tokens):
        chord = tokens[..., :self.n_voices]
        tension = self.tension_table[(chord * self._place).sum(-1)]  # [B, T]
        r = self.tension_weight * inverted_u(tension, self.tension_target, 1.0)
        rest = (chord == 0).float().mean().item()
        distinct = np.mean([len(set(map(tuple, ep.tolist()))) for ep in chord.cpu()])
        return r, {"tension": tension.mean().item(), "rest_frac": rest, "distinct_chords": distinct}

    def render(self, index, sr=22050):
        toks = self.last_tokens[index].cpu().tolist()
        notes, held, time = [], [None] * self.n_voices, 0.0
        for *voices, d in toks:
            dur = self.base_dur * self.dur_ratios[d]
            for v, p in enumerate(voices):
                if p and held[v] is not None and notes[held[v]][2] == self.pitches[p - 1]:
                    s, old, f, a = notes[held[v]]
                    notes[held[v]] = (s, old + dur, f, a)  # same pitch again: sustain
                elif p:
                    notes.append((time, dur, self.pitches[p - 1], 0.25))
                    held[v] = len(notes) - 1
                else:
                    held[v] = None
            time += dur
        return render_notes(notes, self.timbre, sr), sr

    def describe(self):
        steps = ", ".join(f"{cents(s):.0f}" for s in self.scale)
        return (f"Timbre: {self.timbre.name}  partials {[round(r, 3) for r in self.timbre.ratios]}\n"
                f"Ear: {self.ear.name}  s1={self.ear.s1:.4f} s2={self.ear.s2:.2f}\n"
                f"Period (alien octave): ratio {self.period:.4f} = {cents(self.period):.0f} cents\n"
                f"Scale (cents above root): {steps}  -> {len(self.pitches)} pitches over registers\n"
                f"Tension target (normalised): {self.tension_target:.3f}")
