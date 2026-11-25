#!/usr/bin/env bash
# Download the official Efficient-WAM-RT RoboTwin checkpoint (2 GB) into assets/Efficient-WAM-RT/.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://huggingface.co/jiajun0613/Efficient-WAM_RoboTwin/resolve/main/Efficient-WAM-RT
SHA256=209ab4d6f897276633e4d0f36e6b0c573c4938cf0df99d974b05e754ab92340f
OUT=assets/Efficient-WAM-RT/Efficient-WAM-RT_stage3.pt
mkdir -p "$(dirname "$OUT")"
if [[ ! -f $OUT ]]; then
  for _ in $(seq 1 20); do  # resume after dropped connections
    curl -sSL -C - --speed-time 60 --speed-limit 10000 -o "$OUT.part" "$URL/Efficient-WAM-RT_stage3.pt" && break || sleep 5
  done
  mv "$OUT.part" "$OUT"
fi
echo "$SHA256  $OUT" | sha256sum -c -
