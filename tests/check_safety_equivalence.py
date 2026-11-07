"""Optional CPU-only comparison to an external, original 07 delivery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-package", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.reference_only and args.output is None:
        parser.error("--output is required")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch

    torch.set_num_threads(2)
    package = args.original_package.resolve()
    sample_path = package / "example-real.npz"
    weight_path = package / "safety-offline-plus-real.pt"
    with np.load(sample_path, allow_pickle=False) as archive:
        sample = {name: archive[name] for name in archive.files}
    key = "ur5_speedl7_velocity_v1"

    if args.reference_only:
        sys.path.insert(0, str(package))
        from safety_runtime import SafetyScorer
        from safety_verify_wam.portable.adapters import CanonicalSafetyAdapter
        import safety_verify_wam.portable.temporal_risk as module

        assert Path(module.__file__).resolve().is_relative_to(package)
        scorer = SafetyScorer(weight_path, "cpu")
        with torch.inference_mode():
            old_score = scorer.score(key, **sample)
            batch = CanonicalSafetyAdapter(scorer.profiles[key]).from_tensors(
                **sample, pixel_range="uint8"
            ).batch
            out = scorer.model(key, batch)
        assert float(old_score["risk_probability"][0]) == float(out["risk_probability"][0])
        print(json.dumps({
            "step": out["step_risk_probability"][0].tolist(),
            "post": float(out["post_risk_probability"][0]),
            "overall": float(out["risk_probability"][0]),
            "threshold": old_score["threshold"], "model_status": old_score["status"],
            "reference_module": str(Path(module.__file__).resolve()),
        }))
        return

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from safe_wam.contracts import Frame, Observation
    from safe_wam.safety_head import CAMERAS, SafetyHead, _prepare_inputs
    import safety_verify_wam.portable.temporal_risk as module
    import PIL

    assert Path(module.__file__).resolve().is_relative_to(root)
    source = package / "safety-repository/src/safety_verify_wam/portable"
    vendored = root / "src/safety_verify_wam/portable"
    names = sorted(p.name for p in vendored.glob("*.py") if p.name != "__init__.py")
    for name in names:
        assert (source / name).read_bytes() == (vendored / name).read_bytes(), name
    reference = json.loads(subprocess.check_output([
        sys.executable, str(Path(__file__).resolve()), "--original-package", str(package),
        "--reference-only",
    ], text=True))
    obs = Observation(tuple(Frame(
        {name: sample["video"][0, i, j].transpose(1, 2, 0)
         for j, name in enumerate(CAMERAS)}, float(sample["video_timestamps"][0, i]))
        for i in range(4)), sample["state"][0, 0], float(sample["state_timestamps"][0, 0]))
    actions = sample["action"][0]
    prepared = _prepare_inputs(obs, actions)
    np.testing.assert_array_equal(prepared["video"], sample["video"])
    np.testing.assert_array_equal(prepared["state"], sample["state"])
    np.testing.assert_array_equal(prepared["action"], sample["action"])
    model = SafetyHead(weight_path, "cpu")
    result = model.score(obs, actions)
    expected = np.r_[reference["step"], reference["post"], reference["overall"]]
    actual = np.r_[result.step, result.post, result.overall]
    error = float(np.max(np.abs(expected - actual)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert result.threshold == reference["threshold"]
    assert result.model_status == reference["model_status"]
    assert result.step_scores_validated is False
    assert abs(result.overall - max(float(result.step.max()), result.post)) <= 1e-7
    # This check concerns preprocessing only, not synthetic validation accuracy.
    from PIL import Image
    varied = Observation(tuple(Frame({name: np.asarray(Image.fromarray(frame.images[name]).resize(
        (53 + 7 * j, 41 + 3 * i), resample=Image.Resampling.NEAREST)).copy()
        for j, name in enumerate(CAMERAS)}, frame.timestamp_s)
        for i, frame in enumerate(obs.frames)), obs.state, obs.state_timestamp_s)
    resized = _prepare_inputs(varied, actions)["video"]
    for i, frame in enumerate(varied.frames):
        for j, name in enumerate(CAMERAS):
            exact = np.asarray(Image.fromarray(frame.images[name]).convert("RGB").resize(
                (128, 128), resample=Image.Resampling.BILINEAR)).transpose(2, 0, 1)
            np.testing.assert_array_equal(resized[0, i, j], exact)
    def ref(path):
        stat = path.stat()
        return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    report = {
        "schema": "safe_wam_cpu_equivalence/v1", "status": "PASS",
        "checkpoint_ref": ref(weight_path), "example_ref": ref(sample_path),
        "candidate_module": str(Path(module.__file__).resolve()),
        "reference_module": reference["reference_module"],
        "unchanged_numerical_modules": names, "comparison": "direct bytes, no hashes",
        "device": "cpu", "threads": 2, "torch": torch.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__,
        "max_probability_error": error, "atol": 1e-7,
        "risk_overall": result.overall, "risk_post": result.post,
        "risk_step_max": float(result.step.max()), "threshold": result.threshold,
        "example_input_arrays_exact": True, "mixed_resolution_pil_preprocessing_exact": True,
        "robot_commands_sent": False, "training_performed": False,
        "file_hashes_computed": False, "step_scores_validated": False,
        "profile_fingerprint": "Original adapter computes its in-memory profile JSON fingerprint unchanged",
        "validation_scope": "Runtime equivalence only; no UR7e or on-robot validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
