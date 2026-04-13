#!/usr/bin/env python3

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from train_join_model import create_model


class JoinOrderModelWithNorm(nn.Module):

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def verify_export(output_path: str, raw_model: nn.Module, mean: torch.Tensor, std_tensor: torch.Tensor) -> bool:

    all_passed = True

    try:
        loaded = torch.jit.load(output_path)
        print("  Reload test:          PASS")
    except Exception as e:
        print(f"  Reload test:          FAIL ({e})")
        return False


    dummy = torch.randn(1, 48)
    with torch.inference_mode():
        output = loaded(dummy)
        
    if output.shape == (1, 6):
        print("  Shape test (1,6):     PASS")
    else:
        print(f"  Shape test (1,6):     FAIL (got {tuple(output.shape)})")
        all_passed = False


    raw_model.eval()
    x = torch.randn(1, 48)
    with torch.inference_mode():
        scripted_out = loaded(x)
        manual_norm = (x - mean) / std_tensor
        raw_out = raw_model(manual_norm)

    if torch.allclose(scripted_out, raw_out, atol=1e-5):
        print("  Baked standardization: PASS")
    else:
        max_diff = (scripted_out - raw_out).abs().max().item()
        print(f"  Baked standardization: FAIL (max diff={max_diff:.2e})")
        all_passed = False

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Export join order model to TorchScript")
    parser.add_argument("--model-dir", default="ml/join_optimizer/models", help="Model directory")
    args = parser.parse_args()

    stats = torch.load(os.path.join(args.model_dir, "feature_stats.pt"), weights_only=True)
    variant = stats.get("variant", "baseline")

    model = create_model(variant)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "join_order_model_best.pt"), weights_only=True))
    model.eval()

    wrapper = JoinOrderModelWithNorm(model, stats["mean"], stats["std"])
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    output_path = os.path.join(args.model_dir, "join_order_model.pt")
    scripted.save(output_path)
    print(f"Exported TorchScript model (variant={variant}, with baked standardization) to {output_path}")

    print("\nVerification:")
    if not verify_export(output_path, model, stats["mean"], stats["std"]):
        print("\nExport verification FAILED")
        sys.exit(1)
    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
