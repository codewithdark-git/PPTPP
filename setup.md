Setup and running experiments (local)
===================================

This document explains how to set up a local Python environment and run the code/experiments in this repository.

Prerequisites
- Python 3.8+ (3.10 recommended)
- CUDA-capable GPU + matching CUDA/cuDNN for accelerated training (optional but recommended for experiments)
- Git

Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install dependencies

```bash
pip install torch torchvision timm tabulate numpy
```

If you have a `requirements.txt` in your workflow, use:

```bash
pip install -r requirements.txt
```

Preparing datasets
- The repository expects standard image datasets (e.g., ImageNet) for full experiments. For quick demos you can use smaller custom datasets or subset scripts.
- Place datasets in a folder accessible by the data loader or update the dataset path flags passed to `main.py`.

Running examples

- Quick demo (runs a lightweight demo configuration):

```bash
python main.py --demo
```

- Run the full experiment suite used in the paper (long-running):

```bash
python main.py --experiment all
```

- Evaluate a pre-trained model with an off-the-shelf PPT++ schedule (eval-only):

```bash
python main.py --model deit_s --schedule [24,16,8] --compress on --eval-only
```

- Fine-tune with compression enabled:

```bash
python main.py --model lvvit_s --compress on --finetune --epochs 30
```

CLI flags and configuration
- Most runtime behavior (thresholding, schedules, compress on/off, finetune/eval-only) is controlled via CLI flags in `main.py`. Use `python main.py --help` to list available options.

Running on Windows vs Linux
- GPU training commands are the same across OSes; ensure correct CUDA/cuDNN and GPU drivers are installed for Windows.
- On Windows use PowerShell or WSL for a more Unix-like experience when running long experiments.

Tips for reproducibility
- Fix random seeds in configs or add deterministic flags where supported.
- Use the `experiments/` scripts for exact configuration used in the paper (they encapsulate schedules, seeds, and model variants).

Troubleshooting
- If CUDA is not available, add `--device cpu` or set `CUDA_VISIBLE_DEVICES=` to control device selection.
- For dependency version conflicts, create a clean venv and install exact versions; consider exporting `pip freeze` from a working environment.

Additional notes
- This file intentionally covers the local setup and running experiments; high-level design, methods, and performance matrices are described in `README.md`.
