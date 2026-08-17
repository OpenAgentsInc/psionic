#!/usr/bin/env python3
"""Pinned Transformers reference for the Qwen3.8 256x256 vision probe."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


PINNED_TRANSFORMERS_REVISION = "0650ff354501cbdb7cb4138da628cc60f4e0ceed"
SOURCE_SHARD_SHA256 = "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"


def sha256_f32(values: np.ndarray) -> str:
    little_endian = np.asarray(values, dtype="<f4", order="C")
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("transformers_checkout", type=Path)
    args = parser.parse_args()

    revision = subprocess.check_output(
        ["git", "-C", str(args.transformers_checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != PINNED_TRANSFORMERS_REVISION:
        raise RuntimeError(
            f"transformers revision mismatch: expected {PINNED_TRANSFORMERS_REVISION}, got {revision}"
        )
    expected_transformers_source = (args.transformers_checkout / "src" / "transformers").resolve()
    transformers_source = Path(transformers.__file__).resolve().parent
    if transformers_source != expected_transformers_source:
        raise RuntimeError(
            "transformers import source mismatch: "
            f"expected {expected_transformers_source}, got {transformers_source}"
        )
    shard_path = args.model_dir / "model-00001-of-00018.safetensors"
    if not shard_path.is_file():
        raise RuntimeError(f"missing source shard {shard_path}")

    image = np.empty((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            image[y, x] = [x, y, (x + y) // 2]
    processor = AutoImageProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        backend="torchvision",
    )
    processed = processor(images=Image.fromarray(image, mode="RGB"), return_tensors="pt")
    pixel_values = processed["pixel_values"]
    grid_thw = processed["image_grid_thw"]

    config = Qwen3_5Config.from_pretrained(args.model_dir, local_files_only=True).vision_config
    config._attn_implementation = "eager"
    model = Qwen3_5VisionModel(config)
    state = load_file(shard_path, device="cpu")
    vision_state = {
        name.removeprefix("model.visual."): tensor
        for name, tensor in state.items()
        if name.startswith("model.visual.")
    }
    missing, unexpected = model.load_state_dict(vision_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"vision state mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    pixel_values = pixel_values.to(device="cuda", dtype=torch.float32)
    grid_thw = grid_thw.to(device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(pixel_values, grid_thw).pooler_output.float()
    torch.cuda.synchronize()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    embeddings = output.cpu().numpy()
    pixel_values_array = pixel_values.cpu().numpy()

    print(
        json.dumps(
            {
                "schema_version": "psionic.qwen38.vision_transformers_reference.v1",
                "transformers_revision": revision,
                "transformers_version": transformers.__version__,
                "transformers_source": str(transformers_source),
                "torch_version": torch.__version__,
                "source_shard_sha256": SOURCE_SHARD_SHA256,
                "processor_class": processor.__class__.__name__,
                "processor_backend": "torchvision",
                "pixel_values_shape": list(pixel_values_array.shape),
                "pixel_values_sha256": sha256_f32(pixel_values_array),
                "grid_thw": grid_thw.cpu().tolist(),
                "output_shape": list(embeddings.shape),
                "output_sha256": sha256_f32(embeddings),
                "elapsed_ms": elapsed_ms,
                "pixel_values": pixel_values_array.tolist(),
                "embeddings": embeddings.tolist(),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
