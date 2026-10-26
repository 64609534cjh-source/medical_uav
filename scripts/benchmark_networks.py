"""Isolated one-thread CPU inference on the same archived real observations."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import tarfile
import time

import numpy as np
import torch

from medical_uav.agents.networks import build_network
from medical_uav.training.collector import Collector
from medical_uav.training.trainer import load_checkpoint
from medical_uav.training.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = build_network("mlp")
    collector = Collector(8, 0, ("low", "medium", "high"))
    buffer, _ = collector.collect(model, 32)
    collector.close()
    obs, masks = buffer.obs.reshape(-1, 403), buffer.action_mask.reshape(-1, 26)
    np.savez(args.output_dir / "observations.npz", obs=obs.numpy(), masks=masks.numpy())
    # Compare the historical M3 implementation and refactored MLP bit for bit.
    state = load_checkpoint("outputs/m3_mixed_mlp_ppo/seed_4/best.pt")
    with tarfile.open("outputs/m3_mixed_mlp_ppo/seed_4/source.tar.gz") as archive:
        source = archive.extractfile("medical_uav/agents/networks.py").read().decode()
    namespace = {}; exec(compile(source, "historical_m3_networks.py", "exec"), namespace)
    historical = namespace["MLPActorCritic"]()
    historical.load_state_dict(state["model_state_dict"])
    model.load_state_dict(state["model_state_dict"])
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(model(obs), historical(obs)))
        assert all(torch.equal(a, b) for a, b in zip(model.act(obs, masks, True), historical.act(obs, masks, True)))
    measurements = {}
    with torch.inference_mode():
        for architecture, checkpoint in (("mlp", "outputs/m3_mixed_mlp_ppo/seed_4/best.pt"),
                                         ("structured", "outputs/m4_structured_pilot_seed0/best.pt")):
            model = build_network(architecture).eval()
            model.load_state_dict(load_checkpoint(checkpoint)["model_state_dict"])
            measurements[architecture] = {"parameters": sum(p.numel() for p in model.parameters()), "batch": {}}
            for batch in (1, 8, 256):
                x, m = obs[:batch], masks[:batch]
                measurements[architecture]["batch"][str(batch)] = {}
                for name, call in (("forward", lambda: model(x)), ("masked_argmax", lambda: model.act(x, m, True))):
                    for _ in range(50): call()
                    samples = []
                    for _ in range(7):
                        start = time.perf_counter()
                        for _ in range(300): call()
                        samples.append((time.perf_counter() - start) / 300 * 1e6)
                    measurements[architecture]["batch"][str(batch)][name] = {"microseconds_per_batch_median": median(samples),
                                                                                     "repeat_us": samples}
    result = {"threads": 1, "torch": str(torch.__version__), "device": "cpu", "warmups": 50,
              "repeats": 7, "calls_per_repeat": 300, "real_observations": 256,
              "observation_archive_sha256": hashlib.sha256((args.output_dir / "observations.npz").read_bytes()).hexdigest(),
              "historical_mlp_exact_forward_and_act": True, "measurements": measurements,
              "scope": "isolated microbenchmark; no other experiment process active; same observations for both architectures"}
    write_json(args.output_dir / "benchmark.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
