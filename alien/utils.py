import ast
import os
import wave

import numpy as np
import torch


def get_device(pref="auto"):
    """ROCm builds of PyTorch expose AMD GPUs through the torch.cuda API, so 'cuda' covers the 9070 XT."""
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_wav(path, audio, sr):
    a = np.asarray(audio, dtype=np.float64)
    peak = np.abs(a).max()
    if peak > 0:
        a = a / peak * 0.9
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def parse_overrides(pairs):
    """['a=1', 'b=[1,2]', 'ear=human'] -> {'a': 1, 'b': [1, 2], 'ear': 'human'}"""
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out
