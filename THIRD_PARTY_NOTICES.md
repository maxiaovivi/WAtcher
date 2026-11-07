# Third-party and source notices

## Efficient-WAM

Upstream: https://github.com/jiajun613/Efficient-WAM

Pinned revision: `2bd75a8c56acfcd5754b98c7ed313176911ccae0`.

The minimal runtime is vendored for reproducible inference. Its Apache-2.0 license and any included third-party notices are retained with the vendored files. Runtime integration and physical-unit preprocessing changes are documented in `docs/efficient_wam_assets.md`. Model weights have separate distribution requirements and are not included.

## Safety verification runtime

Source: the user-provided `07_safety_offline_plus_real` export and its source overlay. The original Apache-2.0 license is retained with the vendored safety runtime. Its numerical model code is kept separate from WAtcher's input validation and composition layer. See `docs/safety_head.md` for the exact checkpoint interface and limitations.

External Python packages retain their respective licenses. Training data, private experiments, credentials and model weights are excluded from this repository.
