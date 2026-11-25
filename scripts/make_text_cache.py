"""Build the UMT5 instruction cache required by `SimulationEfficientWAMPolicy`.

Run from the root of an Efficient-WAM checkout so that its `data` and
`third_party.wan` packages are importable, for example:

    cd /path/to/Efficient-WAM
    python /path/to/WAtcher/scripts/make_text_cache.py \
        --wan-root /path/to/Wan2.2-TI2V-5B \
        --instruction 'Grab the beige mug, ...' \
        --output /path/to/assets/text/hanging_mug.pt
"""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wan-root", type=Path, required=True,
                        help="WAN directory containing models_t5_umt5-xxl-enc-bf16.pth and google/umt5-xxl")
    parser.add_argument("--instruction", required=True, help="Exact task instruction used at run time")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    sys.path.insert(0, str(Path.cwd()))
    from data.common.text_embeddings import WanT5Embedder
    from safe_wam.simulation_policy import SCENE_PREFIX

    prompt = SCENE_PREFIX + args.instruction
    embeddings = WanT5Embedder(args.wan_root, device=args.device).encode([prompt])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(instruction=args.instruction, scene_prefix=SCENE_PREFIX, prompt=prompt,
                    embeddings=embeddings), args.output)
    print(f"Saved {args.output} with embedding shape {tuple(embeddings[0].shape)}")


if __name__ == "__main__":
    main()
