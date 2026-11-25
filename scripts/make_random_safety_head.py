"""Write a randomly initialized safety head, to smoke-test the pipeline without trained weights.

Its risk scores are meaningless: use it only to check that the loop runs end to end.
    python scripts/make_random_safety_head.py assets/safety/random.pt
"""
import sys
from pathlib import Path

from watcher.safety import random_checkpoint

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    print("saved", random_checkpoint(path))
