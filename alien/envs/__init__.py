from .composer import ComposerEnv
from .rhythm import RhythmEnv
from .signal_game import SignalGameEnv

ENVS = {cls.name: cls for cls in (RhythmEnv, SignalGameEnv, ComposerEnv)}
