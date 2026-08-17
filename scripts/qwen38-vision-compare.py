#!/usr/bin/env python3
"""Compare native Psionic and pinned Transformers Qwen3.8 vision probes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise RuntimeError(f"shape mismatch: {actual.shape} != {expected.shape}")
    difference = actual - expected
    rmse = float(np.sqrt(np.mean(difference * difference)))
    expected_rms = float(np.sqrt(np.mean(expected * expected)))
    flattened_actual = actual.ravel()
    flattened_expected = expected.ravel()
    denominator = float(np.linalg.norm(flattened_actual) * np.linalg.norm(flattened_expected))
    cosine = float(np.dot(flattened_actual, flattened_expected) / denominator)
    absolute = np.abs(difference)
    return {
        "shape": list(actual.shape),
        "rmse": rmse,
        "normalized_rmse": rmse / max(expected_rms, 1e-12),
        "maximum_absolute_error": float(np.max(absolute)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "p999_absolute_error": float(np.quantile(absolute, 0.999)),
        "cosine_similarity": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--psionic-revision", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--driver-version", required=True)
    args = parser.parse_args()

    native = json.loads(args.native.read_text())
    reference = json.loads(args.reference.read_text())
    native_pixels = np.asarray(native.pop("pixel_values"), dtype=np.float32)
    reference_pixels = np.asarray(reference.pop("pixel_values"), dtype=np.float32)
    native_embeddings = np.asarray(native.pop("embeddings"), dtype=np.float32)
    reference_embeddings = np.asarray(reference.pop("embeddings"), dtype=np.float32)
    native_pixels = native_pixels.reshape(reference_pixels.shape)
    processor_parity = comparison(native_pixels, reference_pixels)
    output_parity = comparison(native_embeddings, reference_embeddings)
    processor_passed = (
        processor_parity["normalized_rmse"] <= 1e-7
        and processor_parity["maximum_absolute_error"] <= 1e-6
        and processor_parity["cosine_similarity"] >= 0.999999
    )
    output_passed = (
        output_parity["normalized_rmse"] <= 0.07
        and output_parity["p99_absolute_error"] <= 0.2
        and output_parity["cosine_similarity"] >= 0.997
    )
    report = {
        "schema_version": "psionic.qwen38.vision_parity.v1",
        "status": "implemented_early" if processor_passed and output_passed else "partial",
        "psionic_revision": args.psionic_revision,
        "hardware": {
            "gpu_name": args.gpu_name,
            "driver_version": args.driver_version,
            "idle_query": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits",
            "idle_before_each_process": True,
        },
        "native": native,
        "reference": reference,
        "processor_parity": processor_parity | {"passed": processor_passed},
        "output_parity": output_parity | {"passed": output_passed},
        "raw_probe_sha256": {
            "native": sha256_file(args.native),
            "reference": sha256_file(args.reference),
        },
        "claim_boundary": {
            "media": "one deterministic decoded RGB8 image",
            "dimensions": [256, 256],
            "resize": "not_required",
            "native_backend": "cuda",
            "reference_backend": "transformers_cuda_eager_attention",
            "text_decoder_integration": False,
            "openai_serving": False,
            "video_encoder_parity": False,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if processor_passed and output_passed else 1


if __name__ == "__main__":
    sys.exit(main())
