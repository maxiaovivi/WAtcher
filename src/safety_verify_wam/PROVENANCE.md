# Safety inference source provenance

The nine modules in `portable/` are copied without numerical changes from the
user-supplied safety delivery. They have been checked byte-for-byte against
the original `user_safety_heads_20261106/06_offline/` delivery,
`safety-repository/src/safety_verify_wam/portable/`:

`adapters.py`, `contracts.py`, `guard.py`, `model.py`, `multidomain.py`,
`runtime.py`, `sensor_timing.py`, `sequence_timing.py`, `temporal_risk.py`.

`LICENSE` is the unmodified Apache-2.0 license supplied as `./LICENSE` in the
delivery's `source-overlay.tar.gz` (also present in the companion 07 export).
No `NOTICE` member was present in the checked archive. The two
`__init__.py` files are new minimal initializers: they avoid
the original package's eager training, data-loader and host-model imports.
The public RoboTwin wrapper is new code in `safe_wam/simulation_safety.py`.

Only this inference import closure is included. The training scripts, bundle,
checkpoints, examples containing recorded images, and experiment results are
not redistributed here. No new file hashes were computed. The original
adapter's in-memory profile JSON fingerprint is unchanged. Source byte
comparison and strict checkpoint loading are distinct from hash verification.

The supported trained profile is `robotwin_legacy_target14`. The original
checkpoint schema identifier is retained unchanged, despite its historical
name. No robot-specific velocity adapter is included in this public API.
