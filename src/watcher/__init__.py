"""WAtcher: score every world-action-model action chunk before it reaches the simulator.

    from watcher import Gate, run_episode
    from watcher.policy import EfficientWAMPolicy
    from watcher.safety import SafetyHead
"""
from .gate import Decision, Gate, RiskScores
from .loop import run_episode

__all__ = ["Decision", "Gate", "RiskScores", "run_episode"]
