"""Strict loading of trained network weights."""
from pathlib import Path
import torch

def load_network(model, path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if any(key.startswith("network.") for key in state):
        state = {key[len("network."):]: value for key, value in state.items() if key.startswith("network.")}
    model.load_state_dict(state, strict=True)
    return {"epoch": payload.get("epoch"), "global_step": payload.get("global_step")}
