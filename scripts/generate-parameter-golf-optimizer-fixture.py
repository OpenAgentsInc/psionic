#!/usr/bin/env python3
"""Generate the committed Parameter Golf optimizer parity fixture."""

from __future__ import annotations

import argparse
import ast
import json
import os
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as torch_f


class FunctionalShim:
    """Compatibility shim for newer `train_gpt.py` features on older local torch builds."""

    @staticmethod
    def linear(input_tensor, weight, bias=None):
        return torch_f.linear(input_tensor, weight, bias)

    @staticmethod
    def relu(input_tensor):
        return torch.relu(input_tensor)

    @staticmethod
    def cross_entropy(*args, **kwargs):
        return torch_f.cross_entropy(*args, **kwargs)

    @staticmethod
    def rms_norm(input_tensor, normalized_shape, eps=None):
        if len(tuple(normalized_shape)) != 1:
            raise ValueError("fixture shim only supports last-dimension RMSNorm")
        effective_eps = torch.finfo(input_tensor.dtype).eps if eps is None else eps
        mean_square = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        return input_tensor * torch.rsqrt(mean_square + effective_eps)

    @staticmethod
    def scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=False,
        enable_gqa=False,
    ):
        if enable_gqa and q.size(1) != k.size(1):
            if q.size(1) % k.size(1) != 0:
                raise ValueError("query heads must divide evenly across kv heads")
            repeat = q.size(1) // k.size(1)
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return torch_f.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the committed Parameter Golf optimizer parity fixture"
    )
    parser.add_argument(
        "--parameter-golf-root",
        default=str(Path("~/code/parameter-golf").expanduser()),
        help="Local parameter-golf checkout root.",
    )
    parser.add_argument(
        "--output",
        default="fixtures/parameter_golf/train/parameter_golf_optimizer_fixture.json",
        help="Output fixture path.",
    )
    return parser


def extract_symbols(path: Path) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {
        "dist": dist,
        "F": FunctionalShim,
        "Path": Path,
        "Tensor": torch.Tensor,
        "nn": nn,
        "os": os,
        "torch": torch,
        "uuid": uuid,
    }
    wanted = {
        "Hyperparameters",
        "zeropower_via_newtonschulz5",
        "Muon",
        "CONTROL_TENSOR_NAME_PATTERNS",
        "RMSNorm",
        "CastedLinear",
        "Rotary",
        "apply_rotary_emb",
        "CausalSelfAttention",
        "MLP",
        "Block",
        "GPT",
    }
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
            code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
            exec(code, namespace)
            continue
        if isinstance(node, ast.Assign):
            target_ids = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if target_ids & wanted:
                code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
                exec(code, namespace)
    return {name: namespace[name] for name in wanted}


def baseline_hyperparameters(hyperparameters_cls) -> dict[str, float | int]:
    return {
        "iterations": int(hyperparameters_cls.iterations),
        "warmdown_iters": int(hyperparameters_cls.warmdown_iters),
        "max_wallclock_seconds": float(hyperparameters_cls.max_wallclock_seconds),
        "embed_lr": float(hyperparameters_cls.embed_lr),
        "head_lr": float(hyperparameters_cls.head_lr),
        "tied_embed_lr": float(hyperparameters_cls.tied_embed_lr),
        "matrix_lr": float(hyperparameters_cls.matrix_lr),
        "scalar_lr": float(hyperparameters_cls.scalar_lr),
        "muon_momentum": float(hyperparameters_cls.muon_momentum),
        "muon_backend_steps": int(hyperparameters_cls.muon_backend_steps),
        "muon_momentum_warmup_start": float(
            hyperparameters_cls.muon_momentum_warmup_start
        ),
        "muon_momentum_warmup_steps": int(hyperparameters_cls.muon_momentum_warmup_steps),
        "beta1": float(hyperparameters_cls.beta1),
        "beta2": float(hyperparameters_cls.beta2),
        "adam_eps": float(hyperparameters_cls.adam_eps),
        "grad_clip_norm": float(hyperparameters_cls.grad_clip_norm),
    }


def instantiate_baseline_model(symbols: dict[str, object]):
    hyper = symbols["Hyperparameters"]
    GPT = symbols["GPT"]
    return GPT(
        vocab_size=int(hyper.vocab_size),
        num_layers=int(hyper.num_layers),
        model_dim=int(hyper.model_dim),
        num_heads=int(hyper.num_heads),
        num_kv_heads=int(hyper.num_kv_heads),
        mlp_mult=int(hyper.mlp_mult),
        tie_embeddings=bool(hyper.tie_embeddings),
        tied_embed_init_std=float(hyper.tied_embed_init_std),
        logit_softcap=float(hyper.logit_softcap),
        rope_base=float(hyper.rope_base),
        qk_gain_init=float(hyper.qk_gain_init),
    )


def muon_momentum_at_step(hyperparameters: dict[str, float | int], step: int) -> float:
    warmup_steps = int(hyperparameters["muon_momentum_warmup_steps"])
    frac = min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
    return ((1.0 - frac) * float(hyperparameters["muon_momentum_warmup_start"])) + (
        frac * float(hyperparameters["muon_momentum"])
    )


def learning_rate_multiplier(
    hyperparameters: dict[str, float | int],
    step: int,
    elapsed_ms: float,
    max_wallclock_seconds_override: float | None,
) -> float:
    warmdown_iters = int(hyperparameters["warmdown_iters"])
    if warmdown_iters <= 0:
        return 1.0
    if max_wallclock_seconds_override is None or max_wallclock_seconds_override <= 0.0:
        iterations = int(hyperparameters["iterations"])
        warmdown_start = max(iterations - warmdown_iters, 0)
        if warmdown_start <= step < iterations:
            return max((iterations - step) / max(warmdown_iters, 1), 0.0)
        return 1.0
    max_wallclock_ms = 1000.0 * max_wallclock_seconds_override
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = warmdown_iters * step_ms
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    if remaining_ms <= warmdown_ms:
        return remaining_ms / max(warmdown_ms, 1e-9)
    return 1.0


def main() -> None:
    args = build_parser().parse_args()
    parameter_golf_root = Path(args.parameter_golf_root).expanduser().resolve()
    train_gpt_path = parameter_golf_root / "train_gpt.py"
    symbols = extract_symbols(train_gpt_path)
    hyperparameters = baseline_hyperparameters(symbols["Hyperparameters"])
    control_patterns = list(symbols["CONTROL_TENSOR_NAME_PATTERNS"])

    model = instantiate_baseline_model(symbols)
    block_named_params = list(model.blocks.named_parameters())
    matrix_names = sorted(
        f"blocks.{name}"
        for name, parameter in block_named_params
        if parameter.ndim == 2
        and not any(pattern in name for pattern in control_patterns)
    )
    scalar_names = sorted(
        f"blocks.{name}"
        for name, parameter in block_named_params
        if parameter.ndim < 2 or any(pattern in name for pattern in control_patterns)
    )
    if model.skip_weights.numel() > 0:
        scalar_names.append("skip_weights")
        scalar_names.sort()
    tie_embeddings = bool(symbols["Hyperparameters"].tie_embeddings)
    token_learning_rate = (
        float(hyperparameters["tied_embed_lr"])
        if tie_embeddings
        else float(hyperparameters["embed_lr"])
    )
    head_names = ["lm_head.weight"] if model.lm_head is not None else []

    rows = 3
    cols = 2
    parameter_values = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5]
    gradient_values = [0.6, -0.4, 0.2, -0.1, 0.05, -0.3]
    parameter = nn.Parameter(
        torch.tensor(parameter_values, dtype=torch.float32).reshape(rows, cols)
    )
    parameter.grad = torch.tensor(gradient_values, dtype=torch.float32).reshape(rows, cols)
    optimizer = symbols["Muon"](
        [parameter],
        lr=float(hyperparameters["matrix_lr"]),
        momentum=float(hyperparameters["muon_momentum"]),
        backend_steps=int(hyperparameters["muon_backend_steps"]),
    )
    optimizer.step()
    momentum_buffer = (
        optimizer.state[parameter]["momentum_buffer"].detach().cpu().reshape(-1).tolist()
    )

    schedule_cases = {
        "muon_momentum_cases": [
            {"step": 0, "expected": muon_momentum_at_step(hyperparameters, 0)},
            {"step": 250, "expected": muon_momentum_at_step(hyperparameters, 250)},
            {"step": 500, "expected": muon_momentum_at_step(hyperparameters, 500)},
        ],
        "lr_multiplier_cases": [
            {
                "step": 100,
                "elapsed_ms": 10_000.0,
                "max_wallclock_seconds_override": float(
                    hyperparameters["max_wallclock_seconds"]
                ),
                "expected": learning_rate_multiplier(
                    hyperparameters,
                    100,
                    10_000.0,
                    float(hyperparameters["max_wallclock_seconds"]),
                ),
            },
            {
                "step": 1_000,
                "elapsed_ms": 550_000.0,
                "max_wallclock_seconds_override": float(
                    hyperparameters["max_wallclock_seconds"]
                ),
                "expected": learning_rate_multiplier(
                    hyperparameters,
                    1_000,
                    550_000.0,
                    float(hyperparameters["max_wallclock_seconds"]),
                ),
            },
            {
                "step": 19_000,
                "elapsed_ms": 1_234.0,
                "max_wallclock_seconds_override": None,
                "expected": learning_rate_multiplier(
                    hyperparameters,
                    19_000,
                    1_234.0,
                    None,
                ),
            },
        ],
    }

    fixture = {
        "fixture_id": "parameter_golf_optimizer_v1",
        "hyperparameters": hyperparameters,
        "control_tensor_name_patterns": control_patterns,
        "expected_groups": {
            "token_embedding": ["tok_emb.weight"],
            "matrix": matrix_names,
            "scalar": scalar_names,
            "head": head_names,
            "token_learning_rate": token_learning_rate,
            "matrix_learning_rate": float(hyperparameters["matrix_lr"]),
            "scalar_learning_rate": float(hyperparameters["scalar_lr"]),
            "head_learning_rate": float(hyperparameters["head_lr"]),
        },
        "muon_case": {
            "rows": rows,
            "cols": cols,
            "parameter": parameter_values,
            "gradient": gradient_values,
            "learning_rate": float(hyperparameters["matrix_lr"]),
            "momentum": float(hyperparameters["muon_momentum"]),
            "backend_steps": int(hyperparameters["muon_backend_steps"]),
            "updated_parameter": parameter.detach().cpu().reshape(-1).tolist(),
            "momentum_buffer": momentum_buffer,
        },
        "schedule_cases": schedule_cases,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
