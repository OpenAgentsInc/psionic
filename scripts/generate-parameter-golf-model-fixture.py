#!/usr/bin/env python3
"""Generate the committed Parameter Golf baseline-model parity fixture."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
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
        description="Generate the committed Parameter Golf baseline-model parity fixture"
    )
    parser.add_argument(
        "--parameter-golf-root",
        default=str(Path("~/code/parameter-golf").expanduser()),
        help="Local parameter-golf checkout root.",
    )
    parser.add_argument(
        "--output",
        default="fixtures/parameter_golf/models/parameter_golf_baseline_model_fixture.json",
        help="Output fixture path.",
    )
    return parser


def extract_model_symbols(path: Path) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "nn": nn,
        "F": FunctionalShim,
    }
    wanted = {
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
    return {name: namespace[name] for name in wanted}


def deterministic_values(
    tensor_name: str,
    element_count: int,
    modulus: int,
    centered_offset: int,
    stride: int,
    scale_divisor: float,
) -> list[float]:
    name_seed = sum((index + 1) * byte for index, byte in enumerate(tensor_name.encode("utf-8")))
    values = []
    for index in range(element_count):
        raw = (name_seed + stride * index) % modulus
        centered = raw - centered_offset
        values.append(centered / scale_divisor)
    return values


def assign_deterministic_weights(model: nn.Module, initializer: dict[str, object]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = deterministic_values(
                name,
                parameter.numel(),
                int(initializer["modulus"]),
                int(initializer["centered_offset"]),
                int(initializer["stride"]),
                float(initializer["scale_divisor"]),
            )
            parameter.copy_(
                torch.tensor(values, dtype=parameter.dtype).reshape_as(parameter)
            )


def forward_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.tok_emb(input_ids)
    x = FunctionalShim.rms_norm(x, (x.size(-1),))
    x0 = x
    skips = []
    for layer_index in range(model.num_encoder_layers):
        x = model.blocks[layer_index](x, x0)
        skips.append(x)
    for layer_index in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[layer_index].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + layer_index](x, x0)
    x = model.final_norm(x).reshape(-1, x.size(-1))
    if model.tie_embeddings:
        logits_proj = FunctionalShim.linear(x, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(x)
    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return logits.reshape(input_ids.size(0), input_ids.size(1), -1)


def main() -> None:
    args = build_parser().parse_args()
    parameter_golf_root = Path(args.parameter_golf_root).expanduser().resolve()
    symbols = extract_model_symbols(parameter_golf_root / "train_gpt.py")
    GPT = symbols["GPT"]

    config = {
        "vocab_size": 1024,
        "num_layers": 9,
        "model_dim": 512,
        "num_heads": 8,
        "num_kv_heads": 4,
        "mlp_mult": 2,
        "max_context": 1024,
        "tie_embeddings": True,
        "tied_embed_init_std": 0.005,
        "logit_softcap": 30.0,
        "rope_base": 10000.0,
        "qk_gain_init": 1.5,
    }
    initializer = {
        "modulus": 257,
        "centered_offset": 128,
        "stride": 17,
        "scale_divisor": 2048.0,
    }

    model = GPT(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        model_dim=config["model_dim"],
        num_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        mlp_mult=config["mlp_mult"],
        tie_embeddings=config["tie_embeddings"],
        tied_embed_init_std=config["tied_embed_init_std"],
        logit_softcap=config["logit_softcap"],
        rope_base=config["rope_base"],
        qk_gain_init=config["qk_gain_init"],
    )
    assign_deterministic_weights(model, initializer)
    model.eval()

    input_ids = torch.tensor([[1, 17, 42, 99]], dtype=torch.int64)
    target_ids = torch.tensor([[17, 42, 99, 7]], dtype=torch.int64)
    with torch.no_grad():
        logits = forward_logits(model, input_ids)
        loss = FunctionalShim.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        ).item()

    tensor_shapes = []
    for name, parameter in model.named_parameters():
        tensor_shapes.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
            }
        )
    tensor_shapes.sort(key=lambda item: item["name"])

    fixture = {
        "fixture_id": "parameter_golf_baseline_model_v1",
        "config": config,
        "initializer": initializer,
        "input_ids": input_ids.tolist(),
        "target_ids": target_ids.tolist(),
        "expected_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "expected_tensor_shapes": tensor_shapes,
        "expected_logits": {
            "shape": list(logits.shape),
            "values": logits.reshape(-1).tolist(),
        },
        "expected_loss": float(loss),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
