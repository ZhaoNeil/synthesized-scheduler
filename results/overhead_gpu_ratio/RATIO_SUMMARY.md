# GPU vs CPU inference overhead

Per-decision inference cost of the trained RL models (`*_best.pth`), measured with
[`bench_infer.py`](bench_infer.py) at batch size 1 on an idle agent. GPU: NVIDIA
A30; same torch 2.11.0+cu128 build on both devices.

| Decision | CPU (µs) | GPU (µs) | GPU / CPU |
|---|--:|--:|--:|
| Core assignment (per task) | ~173 | ~444 | 2.57 |
| Time-slice update | ~62 | ~185 | 2.98 |

The Rust↔Python (pyo3) communication cost does not depend on the device. The
models are small, so kernel launch and host–device transfers dominate, and the GPU
is about 2.5–3× slower per decision than the CPU.
