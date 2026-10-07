#!/usr/bin/env python3
"""Controlled per-decision inference microbenchmark: CPU vs A30 GPU.

Replicates the exact EVAL inference ops the scheduler runs per decision, at a
fixed batch of 1 (one task placement / one slice update), on an idle agent -- so
the number of calls, batch size, and scheduling contention are all controlled,
isolating the pure inference overhead the episode runs cannot.

Same torch build for both (only torch.device differs), same trained weights.
Reports per-call thread_time (RL inference compute) and perf_counter (Total,
end-to-end incl. H2D/D2H + GPU sync).

Usage (needs a CUDA-enabled torch, e.g. 2.11.0+cu128 for a machine with a GPU):
    pip install --target /some/dir torch==2.11.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
    PYTHONPATH=/some/dir python3 bench_infer.py <model_dir> <best.pth>
e.g. PYTHONPATH=... python3 bench_infer.py \
        scheds/experimental/scx_rl_exec/py exec_best.pth
"""
import os, sys, time, types
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Stub the Rust-injected scxbus module so q_learning imports standalone.
stub = types.ModuleType("scxbus")
stub.get_latest = lambda: None
sys.modules["scxbus"] = stub

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../scheds/experimental/scx_rl_exec/py")
BEST = sys.argv[2] if len(sys.argv) > 2 else "exec_best.pth"
sys.path.insert(0, MODEL_DIR)

import numpy as np
import torch
from q_learning import DuelingDQN, SliceDQN, N_CORES

torch.set_num_threads(1)
SLICE_CANDS = [32, 174, 390, 1633, 1000000]
INPUT_DIM, HIDDEN = 56, 128
SLICE_INPUT = INPUT_DIM + len(SLICE_CANDS) + 1  # 62
N_ACT = N_CORES

N_ITERS, N_WARM = 4000, 500


def build(device):
    ck = torch.load(os.path.join(MODEL_DIR, BEST), map_location=device)
    pin = ck["policy"]["feature.0.weight"].shape[1]         # placement input dim
    pol = DuelingDQN(pin, HIDDEN, N_ACT).to(device).eval()
    pol.load_state_dict(ck["policy"], strict=False)
    sp = ck["slice_policy"]
    sin = sp["net.0.weight"].shape[1]                        # slice input dim
    sout = sp["net.4.weight"].shape[0]                       # #slice candidates
    slc = SliceDQN(sin, 64, sout).to(device).eval()
    slc.load_state_dict(sp, strict=False)
    return pol, slc, pin, sin


def place_once(pol, state_t, device):
    # Exact eval placement path of q_learning.py at batch_size=1.
    with torch.no_grad():
        q = pol(state_t)
        q_stable = q - q.max(dim=1, keepdim=True).values
        _ = q_stable / 0.5
    assigned = torch.zeros(N_ACT, device=device)
    adjusted = q.squeeze(0) - assigned * 0.5
    return int(adjusted.argmax(0).item())


def slice_once(slc, state_t):
    with torch.no_grad():
        q = slc(state_t)
    return int(q.argmax(dim=1).item())


def bench(device_str):
    device = torch.device(device_str)
    pol, slc, pin, sin = build(device)
    rng = np.random.default_rng(0)
    place_state = rng.standard_normal(pin).astype(np.float32)
    slice_state = rng.standard_normal(sin).astype(np.float32)

    out = {}
    for name, net, st, fn in (
        ("action", pol, place_state, place_once),
        ("slice", slc, slice_state, slice_once),
    ):
        # warmup
        for _ in range(N_WARM):
            state_t = torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
            fn(net, state_t, device) if name == "action" else fn(net, state_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        w0, c0 = time.perf_counter(), time.thread_time()
        for _ in range(N_ITERS):
            state_t = torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
            fn(net, state_t, device) if name == "action" else fn(net, state_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = (time.perf_counter() - w0) / N_ITERS * 1e6   # us/call
        cpu = (time.thread_time() - c0) / N_ITERS * 1e6     # us/call
        out[name] = dict(cpu=cpu, total=wall)
    return out


def main():
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
    print(f"model_dir={MODEL_DIR} best={BEST}")
    print(f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()} gpu={dev_name}")
    print(f"iters={N_ITERS} warmup={N_WARM}\n")
    gpu = bench("cuda")
    cpu = bench("cpu")
    print(f"{'decision':<10}{'device':<7}{'RLinfer us/call':>16}{'Total us/call':>15}")
    for k in ("action", "slice"):
        print(f"{k:<10}{'gpu':<7}{gpu[k]['cpu']:>16.2f}{gpu[k]['total']:>15.2f}")
        print(f"{k:<10}{'cpu':<7}{cpu[k]['cpu']:>16.2f}{cpu[k]['total']:>15.2f}")
    print()
    print(f"{'decision':<34}{'RLinfer(cpu/gpu)':>18}{'Total(cpu/gpu)':>16}")
    labels = {"action": "CPU core assignment request",
              "slice": "Preemption time-slice update"}
    for k in ("action", "slice"):
        ri = cpu[k]['cpu'] / gpu[k]['cpu']
        to = cpu[k]['total'] / gpu[k]['total']
        print(f"{labels[k]:<34}{ri:>18.2f}{to:>16.2f}")


if __name__ == "__main__":
    main()
