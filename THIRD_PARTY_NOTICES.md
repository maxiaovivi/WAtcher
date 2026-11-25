# Sources and licenses

- **Efficient-WAM inference runtime**, `src/watcher/vendor/efficient_wam/`: from
  [jiajun613/Efficient-WAM](https://github.com/jiajun613/Efficient-WAM) at commit
  `2bd75a8c56acfcd5754b98c7ed313176911ccae0` (`inference/robotwin/EfficientWAM`, including `deploy_policy.yml`),
  Apache-2.0 (`LICENSE`), with the Wan2.2 modules under their license (`LICENSE-WAN`). Changes, marked
  `WAtcher:` in the code: `model_loader.build_runtime_from_config` accepts `from_teacher=False` and a
  `text_encoder` (the strict checkpoint load replaces every weight, and instructions are pre-encoded);
  `CompactWANModel.allocate` builds the architecture without loading teacher weights; the T5 import is lazy;
  the unused Wan2.1 VAE and benchmark entry points are removed.
- **Safety head**, `src/watcher/safety.py`: re-implements the inference path of the 06 safety delivery
  (`safety_verify_wam.portable`, Apache-2.0, by the WAtcher authors) with the same parameter names.
- **RoboTwin 2.0** is an external installation and is not vendored; it keeps its own license.

Model weights, datasets and the 300Question-Bench are not part of this repository.
