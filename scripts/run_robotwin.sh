#!/usr/bin/env bash
#SBATCH --job-name=watcher
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%x-%j.out
# Run `watcher-robotwin` from the repo's .venv on a GPU node, directly or through sbatch:
#   sbatch --account=<acct> --partition=<gpu partition> scripts/run_robotwin.sh <watcher-robotwin args>
# Rendering needs the NVIDIA Vulkan driver; SAPIEN finds the host ICD file or provides its own.
set -euo pipefail
for root in "${SLURM_SUBMIT_DIR:-}" "$(dirname "$0")/.."; do  # sbatch runs a copy of this script
  [[ -n $root && -x $root/.venv/bin/watcher-robotwin ]] && cd "$root" && break
done
[[ -x .venv/bin/watcher-robotwin ]] || { echo "Run from the WAtcher repo after scripts/setup_env.sh" >&2; exit 1; }
for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  [[ -f $icd ]] && export VK_ICD_FILENAMES=$icd && break
done
if [[ -z ${__EGL_VENDOR_LIBRARY_FILENAMES:-} ]]; then  # SAPIEN 3.0.0b1 crashes if glvnd has no vendor dir
  for egl in /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json \
      "$(.venv/bin/python -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("sapien").origin))')/vulkan_library/10_nvidia.json"; do
    [[ -f $egl ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$egl && break
  done
fi
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl} PYTHONUNBUFFERED=1
exec .venv/bin/watcher-robotwin "$@"
