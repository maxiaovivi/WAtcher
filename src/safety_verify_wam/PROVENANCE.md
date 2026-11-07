# Safety inference source provenance

The nine modules in `portable/` are copied without numerical changes from the
user-supplied `user_safety_heads_20261106/07_offline_plus_real/` delivery,
`safety-repository/src/safety_verify_wam/portable/`:

`adapters.py`, `contracts.py`, `guard.py`, `model.py`, `multidomain.py`,
`runtime.py`, `sensor_timing.py`, `sequence_timing.py`, `temporal_risk.py`.

`LICENSE` is the unmodified Apache-2.0 license supplied as `./LICENSE` in that
delivery's `source-overlay.tar.gz`. No `NOTICE` member was present in that
archive. The two `__init__.py` files are new minimal initializers: they avoid
the original package's eager training, data-loader and host-model imports.
The public wrapper is new code in `safe_wam/safety_head.py`.

Only this inference import closure is included. The training scripts, bundle,
checkpoints, examples containing recorded images, and experiment results are
not redistributed here. No new file hashes were computed. The original
adapter's in-memory profile JSON fingerprint is unchanged. Source byte
comparison and strict checkpoint loading are distinct from hash verification.

The original runtime's trained profile name remains
`ur5_speedl7_velocity_v1`. It is not renamed to imply UR7e validation.
