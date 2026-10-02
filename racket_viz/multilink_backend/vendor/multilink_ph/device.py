from __future__ import annotations

import torch


def resolve_torch_device(spec: str | torch.device = "auto") -> torch.device:
    """Resolve a user-facing PyTorch device specification.

    ``auto`` selects CUDA when available and otherwise CPU.  Explicit CUDA
    requests fail early with a useful error rather than producing a device
    mismatch later in training.
    """
    if isinstance(spec, torch.device):
        device = spec
    else:
        text = str(spec).strip().lower()
        if text == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(text)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device}' was requested, but torch.cuda.is_available() is False. "
                "Use --device cpu/auto or install a CUDA-enabled PyTorch build."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return device


def device_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = torch.cuda.current_device() if device.index is None else device.index
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:
        name = "CUDA"
    return f"cuda:{index} ({name})"
