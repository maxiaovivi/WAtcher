import json
import sys
import types

import numpy as np
from PIL import Image
import pytest

from safe_wam import cli
from safe_wam.contracts import CAMERAS, RiskScores
from safe_wam.io import load_request


def request_file(tmp_path, external=True):
    frames = []
    for i, t in enumerate((-.3, -.2, -.1, 0.)):
        paths = {}
        for j, name in enumerate(CAMERAS):
            filename = f"{i}_{name}.png"
            Image.new("RGB", (9, 7), (i, j, 31)).save(tmp_path / filename)
            paths[name] = filename
        frames.append({"images": paths, "timestamp_s": t})
    body = dict(schema="safe_wam_request/v1", frames=frames, state=[0] * 6 + [.5], state_timestamp_s=0)
    if external:
        body["actions"] = [[.01] * 6 + [0]] * 16
    p = tmp_path / "request.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p, body


def test_relative_images_and_camera_identity(tmp_path, monkeypatch):
    p, body = request_file(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    obs, actions = load_request(p)
    assert actions == body["actions"]
    assert obs.state.shape == (7,) and len(obs.frames) == 4
    for i, frame in enumerate(obs.frames):
        for j, name in enumerate(CAMERAS):
            np.testing.assert_array_equal(frame.images[name][0, 0], [i, j, 31])


@pytest.mark.parametrize("change", [lambda b: b.update(schema="unknown"),
                                    lambda b: b["frames"][0]["images"].pop("hand2")])
def test_invalid_file_contract(tmp_path, change):
    p, body = request_file(tmp_path)
    change(body)
    p.write_text(json.dumps(body))
    with pytest.raises(ValueError): load_request(p)


def test_missing_image_is_not_replaced(tmp_path):
    p, body = request_file(tmp_path)
    (tmp_path / body["frames"][0]["images"]["hand"]).unlink()
    with pytest.raises(FileNotFoundError): load_request(p)


def test_external_actions_cli_never_loads_wam_and_does_not_overwrite(tmp_path, monkeypatch, capsys):
    p, _ = request_file(tmp_path)
    safety = types.ModuleType("safe_wam.safety_head")
    class FakeHead:
        def __init__(self, *a, **k): pass
        def score(self, obs, a): return RiskScores(np.full(16, .1), .2, .2, .5)
    safety.SafetyHead = FakeHead
    wam = types.ModuleType("safe_wam.efficient_wam")
    def forbidden(*a, **k): pytest.fail("External actions must not instantiate WAM")
    wam.EfficientWAMPolicy = forbidden
    monkeypatch.setitem(sys.modules, "safe_wam.safety_head", safety)
    monkeypatch.setitem(sys.modules, "safe_wam.efficient_wam", wam)
    out = tmp_path / "decision.json"
    monkeypatch.setattr(sys, "argv", ["safe-wam", str(p), "--safety-checkpoint", "unused.pt",
                                     "--wam-config", "unused.json", "--output", str(out)])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ALLOW" and result["robot_commands_sent"] is False
    previous = out.read_bytes()
    assert cli.main() == 2
    failed_output = capsys.readouterr()
    assert "Cannot create output file:" in failed_output.err
    assert failed_output.out == ""
    assert out.read_bytes() == previous
