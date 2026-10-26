"""Strict JSON records and RNG isolation shared by training and evaluation."""

import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None  # absent order types and undefined explained variance
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(clean_json(value), allow_nan=False) + "\n")


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)
