"""Signal game: two agents evolve their own sound language (the 'for fun' one).

A speaker (the PPO policy) sees a hidden meaning, say (shape=2, colour=3), and has msg_len frames
to make a sound: a pitch from the alien scale and a loudness per frame. The sound passes through
a noisy channel heard by an alien ear (log energy in bands with random time jitter). A listener
network must decode the meaning from what it hears.

  speaker reward  = log p_listener(true meaning) - energy_weight * loudness
  listener        = trained by plain supervised learning on what it heard

Nobody tells them what a 'call' should sound like. Noise forces redundancy, the energy cost forces
economy, and compositional meanings let structure emerge, roughly the pressures that shaped birdsong.
Run render.py on a checkpoint to hear the whole lexicon: one call per meaning.
"""
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sound import derive_scale, make_ear, make_timbre, render_notes


class SignalGameEnv:
    name = "signal"
    ppo_defaults = {"episodes_per_iter": 2, "minibatch": 1024}

    def __init__(self, num_envs=512, n_attrs=2, n_values=4, msg_len=8, n_pitches=12, n_loud=4, n_bands=48,
                 hearing=(60.0, 9000.0), band_width=0.15, noise=0.3, max_shift=1, energy_weight=0.1,
                 listener_hidden=256, listener_lr=1e-3, listener_steps=2, frame_s=0.11, f_base=150.0,
                 ear="alien", timbre="alien", seed=0, device="cpu"):
        self.device = torch.device(device)
        self.num_envs, self.n_attrs, self.n_values, self.msg_len = num_envs, n_attrs, n_values, msg_len
        self.n_loud, self.noise, self.max_shift = n_loud, noise, max_shift
        self.energy_weight, self.listener_steps, self.frame_s = energy_weight, listener_steps, frame_s
        self.timbre, self.ear = make_timbre(timbre, seed), make_ear(ear, seed)
        self.scale, self.period = derive_scale(self.timbre, self.ear, f0=f_base * 2)
        self.pitches = [f_base * self.period ** k * s for k in range(8) for s in self.scale][:n_pitches]
        self.n_pitches = len(self.pitches)

        # What the listener's ear hears per pitch: partial energy smeared over log-spaced bands.
        log_c = np.log(np.geomspace(*hearing, n_bands))
        spec = np.zeros((self.n_pitches, n_bands))
        for i, f0 in enumerate(self.pitches):
            for r, a in zip(self.timbre.ratios, self.timbre.amps):
                spec[i] += a * np.exp(-0.5 * ((np.log(f0 * r) - log_c) / band_width) ** 2)
        self.spec_table = torch.tensor(spec, dtype=torch.float32, device=self.device)

        self.listener = nn.Sequential(
            nn.Linear(msg_len * n_bands, listener_hidden), nn.GELU(),
            nn.Linear(listener_hidden, listener_hidden), nn.GELU(),
            nn.Linear(listener_hidden, n_attrs * n_values)).to(self.device)
        self.listener_opt = torch.optim.Adam(self.listener.parameters(), lr=listener_lr)

        self.actions = torch.zeros(num_envs, msg_len, 2, dtype=torch.long, device=self.device)
        self.meanings = None
        self.last_actions = self.last_meanings = None
        self.learn = True  # False = don't train the listener (generation only)
        self.t = 0

    @property
    def episode_len(self):
        return self.msg_len

    @property
    def obs_dim(self):
        return self.n_attrs * self.n_values + self.msg_len * (self.n_pitches + self.n_loud) + self.msg_len

    @property
    def action_dims(self):
        return [self.n_pitches, self.n_loud]

    def all_meanings(self):
        return torch.tensor(list(itertools.product(range(self.n_values), repeat=self.n_attrs)), device=self.device)

    def _obs(self):
        B, L = self.num_envs, self.msg_len
        meaning = F.one_hot(self.meanings, self.n_values).float().flatten(1)
        hist = torch.cat([F.one_hot(self.actions[..., 0], self.n_pitches),
                          F.one_hot(self.actions[..., 1], self.n_loud)], -1).float()
        hist = hist * (torch.arange(L, device=self.device) < self.t)[None, :, None]
        pos = F.one_hot(torch.full((B,), min(self.t, L - 1), device=self.device), L).float()
        return torch.cat([meaning, hist.flatten(1), pos], -1)

    def reset(self, meanings=None):
        if meanings is None:
            meanings = torch.randint(self.n_values, (self.num_envs, self.n_attrs), device=self.device)
        self.meanings = meanings.to(self.device)
        self.actions.zero_()
        self.t = 0
        return self._obs()

    def _channel(self, actions):
        B, L = actions.shape[:2]
        loud = actions[..., 1].float() / (self.n_loud - 1)
        spec = self.spec_table[actions[..., 0]] * loud.unsqueeze(-1)
        x = torch.log(spec + self.noise * torch.rand_like(spec) + 1e-2)
        if self.max_shift:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (B, 1), device=self.device)
            idx = (torch.arange(L, device=self.device)[None] - shift) % L
            x = x.gather(1, idx.unsqueeze(-1).expand_as(x))
        return x.flatten(1), loud

    def _decode_loss(self, x, meanings):
        logits = self.listener(x).view(-1, self.n_attrs, self.n_values)
        return F.cross_entropy(logits.transpose(1, 2), meanings, reduction="none"), logits

    def step(self, action):
        self.actions[:, self.t] = action
        self.t += 1
        done = self.t == self.msg_len
        info = {}
        if done:
            x, loud = self._channel(self.actions)
            with torch.no_grad():
                nll, logits = self._decode_loss(x, self.meanings)
            energy = loud.mean(1)
            reward = -nll.mean(1) - self.energy_weight * energy
            for _ in range(self.listener_steps if self.learn else 0):
                self.listener_opt.zero_grad()
                self._decode_loss(x, self.meanings)[0].mean().backward()
                self.listener_opt.step()
            retro = torch.zeros(self.msg_len, self.num_envs, device=self.device)
            retro[-1] = reward
            msgs = {tuple(m) for m in self.actions.flatten(1).tolist()}
            info["retro_reward"] = retro
            info["metrics"] = {"listener_acc": (logits.argmax(-1) == self.meanings).float().mean().item(),
                               "energy": energy.mean().item(),
                               "distinct_msgs": len(msgs) / self.num_envs}
            self.last_actions, self.last_meanings = self.actions.clone(), self.meanings.clone()
        return self._obs(), torch.zeros(self.num_envs, device=self.device), done, info

    def message_notes(self, actions, start=0.0):
        notes = []
        for t, (p, l) in enumerate(actions.tolist()):
            if l == 0:
                continue
            amp = 0.3 * l / (self.n_loud - 1)
            prev = notes[-1] if notes else None
            if prev and prev[2] == self.pitches[p] and abs(prev[0] + prev[1] - (start + t * self.frame_s)) < 1e-6:
                notes[-1] = (prev[0], prev[1] + self.frame_s, prev[2], prev[3])
            else:
                notes.append((start + t * self.frame_s, self.frame_s, self.pitches[p], amp))
        return notes

    def render(self, index, sr=22050):
        return render_notes(self.message_notes(self.last_actions[index]), self.timbre, sr), sr

    def render_lexicon(self, actions, sr=22050, gap=0.5):
        """actions [N, L, 2], one message per meaning, played in order with gaps."""
        notes, span = [], self.msg_len * self.frame_s + gap
        for i, a in enumerate(actions):
            notes += self.message_notes(a, start=i * span)
        return render_notes(notes, self.timbre, sr), sr

    def describe(self):
        return (f"Meanings: {self.n_attrs} attributes x {self.n_values} values = {self.n_values ** self.n_attrs}\n"
                f"Timbre: {self.timbre.name}, ear: {self.ear.name}\n"
                f"Pitches (Hz): {[round(p, 1) for p in self.pitches]}")

    def state_dict(self):
        return {"listener": self.listener.state_dict(), "listener_opt": self.listener_opt.state_dict()}

    def load_state_dict(self, sd):
        self.listener.load_state_dict(sd["listener"])
        self.listener_opt.load_state_dict(sd["listener_opt"])
