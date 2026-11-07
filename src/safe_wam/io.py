"""Local JSON observation files. Image paths are resolved relative to that file."""
import json
from pathlib import Path

import numpy as np

from .contracts import CAMERAS, Frame, Observation


def load_request(path: str | Path):
    from PIL import Image
    path = Path(path).resolve()
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("schema") != "safe_wam_request/v1":
        raise ValueError("Expected schema safe_wam_request/v1")
    frames = []
    for frame in request["frames"]:
        if set(frame["images"]) != set(CAMERAS):
            raise ValueError("Each frame must contain exactly the three named cameras")
        images = {}
        for name in CAMERAS:
            with Image.open(path.parent / frame["images"][name]) as image:
                images[name] = np.asarray(image.convert("RGB")).copy()
        frames.append(Frame(images, float(frame["timestamp_s"])))
    observation = Observation(tuple(frames), np.asarray(request["state"], dtype=np.float32),
                              float(request["state_timestamp_s"]))
    return observation, request.get("actions")
