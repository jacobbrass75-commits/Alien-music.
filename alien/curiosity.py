"""Compression-progress reward (Schmidhuber-style curiosity).

A separate predictor learns to predict the next event from the previous `window` events.
The composer is rewarded by how much the predictor *improved* on its piece during one update:
    reward_t = loss_before_t - loss_after_t
Silence/repetition is learned once and then pays nothing; noise never becomes predictable and
pays nothing. Only learnable-but-not-yet-learned structure pays, so the reward is an
inverted U by construction and keeps moving as the predictor gets better.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot_tokens(tokens, head_sizes):
    """tokens [..., H] long (-1 = empty) -> [..., sum(head_sizes)] float."""
    parts = []
    for h, s in enumerate(head_sizes):
        tok = tokens[..., h]
        parts.append(F.one_hot(tok.clamp(min=0), s).float() * (tok >= 0).unsqueeze(-1).float())
    return torch.cat(parts, -1)


def context_windows(tokens, head_sizes, window):
    """tokens [B, T, H] -> [B, T, window*D]: for each t, the one-hot events t-window..t-1."""
    oh = onehot_tokens(tokens, head_sizes)
    B, T, D = oh.shape
    padded = F.pad(oh, (0, 0, window, 0))
    win = padded.unfold(1, window, 1)[:, :T]  # [B, T, D, window]
    return win.permute(0, 1, 3, 2).reshape(B, T, window * D)


class TokenPredictor(nn.Module):
    def __init__(self, head_sizes, window, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(window * sum(head_sizes), hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.heads = nn.ModuleList(nn.Linear(hidden, s) for s in head_sizes)

    def forward(self, x):
        h = self.net(x)
        return [head(h) for head in self.heads]


class LearningProgress:
    def __init__(self, head_sizes, window, device, hidden=256, lr=1e-3, inner_steps=2, ema=0.98):
        self.head_sizes, self.window = head_sizes, window
        self.model = TokenPredictor(head_sizes, window, hidden).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.inner_steps, self.ema = inner_steps, ema
        self.scale = None

    def _loss(self, ctx, tgt):
        return sum(F.cross_entropy(logits, tgt[:, h], reduction="none")
                   for h, logits in enumerate(self.model(ctx)))

    def __call__(self, tokens):
        """tokens [B, T, H] complete episodes -> (normalised progress [B, T], surprisal [B, T], metrics).

        surprisal is the predictor's loss in nats *before* it trained on this batch."""
        B, T, H = tokens.shape
        ctx = context_windows(tokens, self.head_sizes, self.window).reshape(B * T, -1)
        tgt = tokens.reshape(B * T, H)
        with torch.no_grad():
            before = self._loss(ctx, tgt)
        for _ in range(self.inner_steps):
            self.opt.zero_grad()
            self._loss(ctx, tgt).mean().backward()
            self.opt.step()
        with torch.no_grad():
            after = self._loss(ctx, tgt)
        lp = (before - after).view(B, T)
        # Normalise by a running scale so lp_weight is comparable to the other reward terms.
        m = lp.abs().mean().item()
        self.scale = m if self.scale is None else self.ema * self.scale + (1 - self.ema) * m
        return (lp / max(self.scale, 1e-3), before.view(B, T),
                {"pred_loss": after.mean().item(), "lp_raw": lp.mean().item()})

    def state_dict(self):
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "scale": self.scale}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.scale = sd["scale"]
