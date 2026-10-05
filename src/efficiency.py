"""Parameter count, conv/linear FLOPs, and single-image latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from src.model import SareeEncoder


def device_of() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_conv_linear_flops(model: nn.Module, device: torch.device) -> int:
    total = 0

    def conv_hook(module, inputs, output):
        nonlocal total
        n, _, h, w = output.shape
        kh, kw = module.kernel_size
        total += int(2 * n * module.in_channels * kh * kw * h * w * module.out_channels / module.groups)

    def linear_hook(module, inputs, output):
        nonlocal total
        batch = output.shape[0]
        total += int(2 * batch * module.in_features * module.out_features)

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224, device=device))
    for handle in handles:
        handle.remove()
    return total


def latency_ms(model: nn.Module, device: torch.device, iters: int = 100) -> float:
    sample = torch.zeros(1, 3, 224, 224, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (torch.mps.synchronize if device.type == "mps" else lambda: None)
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        sync()
        start = time.perf_counter()
        for _ in range(iters):
            model(sample)
        sync()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/color/best.pt"))
    args = parser.parse_args()

    device = device_of()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SareeEncoder(embed_dim=saved["args"]["embed_dim"], pretrained=False)
    model.load_state_dict(saved["model"])
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    flops = count_conv_linear_flops(model, device)
    report = {
        "backbone": "torchvision MobileNetV3-Small, ImageNet-1K weights, then fine-tuned",
        "parameters": params,
        "parameters_millions": round(params / 1e6, 3),
        "conv_linear_flops": flops,
        "conv_linear_gflops": round(flops / 1e9, 3),
        "embedding_dim": model.embed_dim,
        "input": "1x3x224x224",
        "latency_ms_batch1": round(latency_ms(model, device), 2),
        "device": str(device),
        "notes": "FLOPs count multiply-adds in Conv2d and Linear only.",
    }
    out = args.checkpoint.parent / "efficiency.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
