from __future__ import annotations

from pathlib import Path
from typing import Any
import torch


def save_stage_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    history: list,
    best_state: dict[str, torch.Tensor] | None = None,
    best_value: float | None = None,
    best_metrics: dict | None = None,
    rng_state: object | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch),
        "history": history,
        "best_state": best_state,
        "best_value": best_value,
        "best_metrics": best_metrics or {},
        "rng_state": rng_state,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_stage_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    state = payload.get("model_state", payload.get("state_dict"))
    if state is None:
        raise KeyError(f"checkpoint {path} has no model_state/state_dict")
    model.load_state_dict(state)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move tensor-valued optimizer state after loading a checkpoint."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
