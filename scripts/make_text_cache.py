"""Precompute the UMT5 embedding of a task instruction (WAtcher never runs the text encoder online).

    python scripts/make_text_cache.py --wan-root /path/to/Wan2.2-TI2V-5B \
        --instruction 'Grab the beige mug, ...' --output assets/text/hanging_mug.pt
"""
import argparse
from pathlib import Path

import torch

from watcher.policy import SCENE_PREFIX
from watcher.vendor.efficient_wam.third_party.wan.modules.t5 import T5EncoderModel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wan-root", type=Path, required=True,
                   help="folder with models_t5_umt5-xxl-enc-bf16.pth and google/umt5-xxl")
    p.add_argument("--instruction", required=True, help="exact instruction used at run time")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"{a.output} exists")
    encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device(a.device),
                             checkpoint_path=str(a.wan_root / "models_t5_umt5-xxl-enc-bf16.pth"),
                             tokenizer_path=str(a.wan_root / "google" / "umt5-xxl"))
    prompt = SCENE_PREFIX + a.instruction
    with torch.inference_mode():
        embeddings = [e.detach().cpu() for e in encoder([prompt], torch.device(a.device))]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(instruction=a.instruction, scene_prefix=SCENE_PREFIX, prompt=prompt, embeddings=embeddings),
               a.output)
    print(f"saved {a.output}: embedding {tuple(embeddings[0].shape)}")


if __name__ == "__main__":
    main()
