"""Optional original-06 CPU check on one saved, real RoboTwin training input.

No labels are used, no simulator is run, and no checkpoint is trained.
--prepared points to an existing trusted tensor archive, not a download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-package", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
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
    checkpoint = package / "safety-offline.pt"
    # The existing archive is trusted; select by recorded training provenance,
    # never labels or predictions. Only the public model-input tensors are used.
    data = torch.load(args.prepared, map_location="cpu", weights_only=False)
    row = next(row for row in data["rows"] if row["split"] == "train"
               and row["source_generation"] == "old_regression" and row["environment"] == "robotwin")
    sample = data["samples"][row["case_id"]]
    assert tuple(sample["video"].shape) == (1, 3, 3, 128, 128)
    assert tuple(sample["state"].shape) == (1, 14)
    assert tuple(sample["action"].shape) == (16, 14)
    assert not bool(sample["action_time_known"].any()) and sample["action_dt"] is None
    assert not bool(sample["action_timestamps"].any())
    inputs = {name: sample[name][None] for name in
              ("video", "state", "action", "video_timestamps", "state_timestamps")}
    key = "robotwin_legacy_target14"
    if args.reference_only:
        sys.path.insert(0, str(package))
        from safety_runtime import SafetyScorer
        from safety_verify_wam.portable.adapters import CanonicalSafetyAdapter
        import safety_verify_wam.portable.temporal_risk as module

        assert Path(module.__file__).resolve().is_relative_to(package)
        model = SafetyScorer(checkpoint, "cpu")
        with torch.inference_mode():
            runtime_score = model.score(key, **inputs)
            batch = CanonicalSafetyAdapter(model.profiles[key]).from_tensors(
                **inputs, pixel_range="uint8", action_timestamps=None).batch
            out = model.model(key, batch)
        assert float(runtime_score["risk_probability"][0]) == float(out["risk_probability"][0])
        print(json.dumps(dict(step=out["step_risk_probability"][0].tolist(),
                              post=float(out["post_risk_probability"][0]),
                              overall=float(out["risk_probability"][0]),
                              threshold=runtime_score["threshold"], status=runtime_score["status"],
                              module=str(Path(module.__file__).resolve()))))
        return

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from safe_wam.simulation_safety import CAMERAS, SimulationObservation, SimulationSafetyHead, prepare_simulation_inputs
    import safety_verify_wam.portable.temporal_risk as module
    import PIL

    assert Path(module.__file__).resolve().is_relative_to(root)
    source = package / "safety-repository/src/safety_verify_wam/portable"
    vendored = root / "src/safety_verify_wam/portable"
    modules = sorted(p.name for p in vendored.glob("*.py") if p.name != "__init__.py")
    for name in modules:
        assert (source / name).read_bytes() == (vendored / name).read_bytes(), name
    reference = json.loads(subprocess.check_output([
        sys.executable, str(Path(__file__).resolve()), "--original-package", str(package),
        "--prepared", str(args.prepared), "--reference-only",
    ], text=True))
    observation = SimulationObservation(
        {name: sample["video"][0, j].permute(1, 2, 0).numpy() for j, name in enumerate(CAMERAS)},
        sample["state"][0].numpy(), float(sample["video_timestamps"][0]), float(sample["state_timestamps"][0]),
    )
    actions = sample["action"].numpy()
    prepared = prepare_simulation_inputs(observation, actions)
    for name in inputs:
        np.testing.assert_array_equal(prepared[name], inputs[name].numpy())
    model = SimulationSafetyHead(checkpoint, "cpu")
    result = model.score(observation, actions)
    with torch.inference_mode():
        batch = model._adapter.from_tensors(**prepared, pixel_range="uint8", action_timestamps=None).batch
    assert not bool(batch.action_time_known.any())
    # SafetyBatch represents unknown times with zero timestamps + false mask;
    # the original profile, not SafetyBatch, owns control_dt.
    assert not bool(batch.action_timestamps.any()) and model.profile.control_dt is None
    np.testing.assert_array_equal(batch.state.numpy(),
        ((inputs["state"].float() - torch.tensor(model.profile.state_mean, dtype=torch.float32))
         / torch.tensor(model.profile.state_std, dtype=torch.float32)).numpy())
    np.testing.assert_array_equal(batch.action.numpy(),
        ((inputs["action"].float() - torch.tensor(model.profile.action_mean, dtype=torch.float32))
         / torch.tensor(model.profile.action_std, dtype=torch.float32)).numpy())
    expected = np.r_[reference["step"], reference["post"], reference["overall"]]
    actual = np.r_[result.step, result.post, result.overall]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert result.threshold == reference["threshold"] and result.model_status == reference["status"]
    assert abs(result.overall - max(float(result.step.max()), result.post)) < 1e-7
    def ref(path):
        path = Path(path)
        stat = path.stat()
        return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    receipt = dict(
        schema="watcher_simulation_cpu_equivalence/v1", status="PASS",
        original_package=str(package), checkpoint_ref=ref(checkpoint), prepared_ref=ref(args.prepared),
        selected_source_case_id=row["case_id"], selected_source_generation=row["source_generation"],
        selection="First recorded old_regression Robotwin train row; no label/score selection",
        source_reference_module=reference["module"], candidate_module=str(Path(module.__file__).resolve()),
        unchanged_numerical_modules=modules, source_comparison="direct bytes, not hashes",
        original_inputs_exact=True, normalization_exact=True, action_timing_unknown_preserved=True,
        outputs_compared=18, max_absolute_error=float(np.abs(actual-expected).max()), atol=1e-7,
        overall=result.overall, post=result.post, maximum_step=float(result.step.max()),
        threshold=result.threshold, step_scores_validated=False,
        torch=torch.__version__, numpy=np.__version__, pillow=PIL.__version__, device="cpu", threads=2,
        labels_used=False, file_hashes_computed=False, simulation_performed=False,
        training_performed=False, robot_commands_sent=False,
        validation_scope="Inference equivalence only; no safety accuracy or execution protocol validation",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
