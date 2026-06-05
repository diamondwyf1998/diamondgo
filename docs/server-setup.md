# AutoDL Server Setup

Target rental for the first real run:

- GPU: RTX 4090D
- CPU: 16-core Xeon Platinum 8352S class
- Rental duration: 1 day

## Recommended Image

Prefer an AutoDL image that already includes:

- Ubuntu 22.04 or 20.04;
- NVIDIA driver compatible with 4090D;
- CUDA 12.1 or 12.4;
- Python 3.10 or 3.11;
- PyTorch 2.x with CUDA enabled;
- JupyterLab or SSH access.

If AutoDL offers a PyTorch image, choose that over a bare CUDA image.

## Project Dependencies

Install after connecting:

```bash
git clone <your-repo-url> diamondgo
cd diamondgo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the first full rules run, `sgfmill` is needed through the project
dependencies. For the current CPU smoke demo, the fallback simplified rules
backend can run without it.

## Checks Before Training

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## First GPU Run Shape

Start conservatively:

- board size: 9
- network: 64 channels, 4 residual blocks
- self-play games: 16 to 64
- MCTS simulations: 64
- batch size: 128 or 256
- learning rate: 1e-3
- replay buffer: keep small until SGF/search diagnostics look sane

On a 4090D, the bottleneck will quickly become Python-side self-play and MCTS
unless we batch network inference. Do not overinterpret GPU utilization during
the first run.
