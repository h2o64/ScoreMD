<h1 align="center">Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models</h1>
<p align="center">
<a href="https://arxiv.org/abs/2506.17139"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
<a href="https://papers.nips.cc/paper_files/paper/2025/hash/231be94eaf8dcbc49a95b256c9b6b8b5-Abstract-Conference.html"><img src="https://img.shields.io/badge/NeurIPS-4f2d4e?style=for-the-badge" alt="NeurIPS"/></a>
<a href="https://colab.research.google.com/drive/1r3DGOpGZgIbx7p_uWbsm0b6iP-kiyU_v"><img src="https://img.shields.io/badge/Colab jax-e37e3d.svg?style=for-the-badge&logo=googlecolab&logoColor=white" alt="JAX Colab"/></a>
<a href="https://colab.research.google.com/drive/1rbcNDwDduPw5QMulbOC-Pg6xGd4wGjej"><img src="https://img.shields.io/badge/Colab PyTorch-e37e3d.svg?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab PyTorch"/></a>
</p>

<p align="center">
  <img src="images/anim.gif" style="width: 100%; height: auto; max-width: 100%;"/>
</p>

<p align="center">
  <em>Animation showing the two modes of our model: independent sampling by diffusion denoising (left) and molecular dynamics simulation (right).</em>
</p>

> [!NOTE]
> **This is a research fork.** This codebase has been extended for the paper
> [*A Diffusive Classification Loss for Learning Energy-based Generative Models*](https://arxiv.org/abs/2601.21025) (ICML 2026).
> The original ScoreMD pipeline is unchanged; the additions are opt-in through new Hydra flags. See [Paper modifications](#paper-modifications) below.

## Overview

This repository contains the complete codebase for training and evaluating energy-based diffusion models for molecular dynamics simulations. Our approach enables a single model to perform both independent sampling via diffusion denoising and continuous molecular dynamics simulations through a Fokker-Planck-based regularization scheme.

## Method

<p align="center">
<img src="images/main.png" alt="Visualization of the main idea of this paper."/>
</p>

<p align="center">
  <em>We introduce a Fokker-Planck-based regularization to train an energy-based diffusion model with stable, self-consistent scores near the data distribution. This regularization ensures that the learned score function corresponds to a consistent energy function, enabling the model to perform both generative sampling and accurate energy-estimation.</em>
</p>

## Tutorial

We provide minimal working implementations in Jupyter notebooks:
- **[JAX version](https://colab.research.google.com/drive/1r3DGOpGZgIbx7p_uWbsm0b6iP-kiyU_v)** (recommended) - Faster and closer to this repository's implementation
- **[PyTorch version](https://colab.research.google.com/drive/1rbcNDwDduPw5QMulbOC-Pg6xGd4wGjej)** - Alternative implementation

Both notebooks demonstrate how to reproduce similar figures to the one shown above and perform molecular dynamics simulations with a diffusion model on the Müller-Brown potential. 

# 🚀 Getting Started

## Installation

### pixi
All dependencies are managed with [pixi](https://github.com/prefix-dev/pixi), which ensures fully reproducible environments across different systems.

To set up the environment, run:

```bash
pixi install --frozen
```


To activate the environment, run:
```bash
pixi shell
```

### Docker
If you are on an amd64 system (e.g. a Linux machine), you can use the docker image to run the code. To build the docker image, run:

```bash
docker build -t scoremd .
```

To run the docker container, run:

```bash
docker run -it --rm -v $(pwd)/outputs:/workspace/outputs -v $(pwd)/storage:/workspace/storage -v $(pwd)/multirun:/workspace/multirun scoremd python train.py ...
```

### Alternative Installation Methods

If you prefer using your own dependency manager (e.g., conda, pip), you can install the dependencies listed in `pyproject.toml` with your preferred tool.


## Quick Start: Toy Systems

We use [Hydra](https://hydra.cc/) for configuration management. You can override any configuration via command-line arguments or configuration files.

Train on example toy systems using the provided configurations:

```bash
python train.py dataset=double_well +architecture=mlp/small_potential
python train.py dataset=double_well_2d +architecture=mlp/small_potential
```

Outputs will be saved to the `outputs/` directory.

# 🧬 Working with Molecules
> [!IMPORTANT]
> This repository does **not** contain all datasets directly.
> Training data for the toy systems and alanine dipeptide will be downloaded automatically. 
> For the dipeptides, you can download the dataset from [this release](https://github.com/noegroup/ScoreMD/releases/tag/1.0.0) and place it into the `./storage` directory (one subfolder for each dataset, e.g. `./storage/minipeptides/`, `./storage/deshaw/`).
> Data for the fast-folder systems can be requested from D. E. Shaw Research, as described in the [original paper](https://www.science.org/doi/full/10.1126/science.1208351).
> If you do not have access to the fast-folder data, [this release](https://github.com/noegroup/ScoreMD/releases/tag/1.0.0) also provides dummy data generated by our models, which is sufficient for inference.

## Running Inference with Pre-Trained Models

We provide pre-trained model weights for all models presented in the paper. For detailed instructions on downloading and using these models, please refer to [INFERENCE.md](INFERENCE.md).

## Training Models from the Paper

To reproduce the results from our paper, see [TRAIN.md](TRAIN.md) for the exact training commands used for each model and dataset. 

## Evaluation and Plotting Scripts

For implementation details and benchmarking against your own methods, we provide evaluation scripts in the [evaluation](evaluation/README.md) directory.

# Paper modifications

This fork adds the auxiliary losses, samplers, and evaluation hooks used in
[*A Diffusive Classification Loss for Learning Energy-based Generative Models*](https://arxiv.org/abs/2601.21025) (ICML 2026).
All additions are off by default; the original training and evaluation paths are unchanged.

## What's new

- **DiffCLF loss** (`src/scoremd/diffusion/diffclf.py`, `time_sampling.py`) — a contrastive
  classification objective over noise levels. The `k=2` binary form is the loss used in the paper;
  `k>2` falls back to a multi-class form.
- **RNE loss** (`src/scoremd/diffusion/rne.py`) — residual-noise-estimation regularizer that
  matches forward/backward transition densities of the diffusion. Needs `VP.forward_transition_params`
  / `VP.backward_transition_params`, added in `src/scoremd/diffusion/classic/sde.py`.
- **DSM warmup epochs** (`src/scoremd/training/schedule.py`) — option to run the first N epochs
  with the plain score-matching loss before turning on DiffCLF/RNE.
- **MH-corrected reverse-time sampling** (`src/scoremd/diffusion/classic/solvers.py:EulerMaruyamaWithMH`,
  `get_sampler(collect_mh_accept_rate=True)`, `evaluate.py` helpers) — Metropolis–Hastings accept/reject
  step at each reverse-time index, using the model's `log_q` as the target.
- **HMC MD step** (`src/scoremd/simulation.py:step_with_mh`) — leapfrog HMC alternative to the
  plain Langevin integrator at evaluation time, threaded through the dataset classes via `with_mh`.
- **Optional time-dependent `log_Z`** (`src/scoremd/models/timenet.py`, `GraphTransformer.log_Z`) —
  a small sinusoidal-embedding MLP subtracted from `log_q(x, t)` to absorb the time-varying
  normalizer.

## How to use

Activate via Hydra overrides on top of the existing training/evaluation commands.

**DiffCLF training (paper setup, ALDP):**
```bash
python train.py dataset=aldp +architecture=transformer/potential \
  dataset.coarse_graining_level=full \
  training_schedule.losses.0.loss.alpha=0.0 \
  training_schedule.losses.0.loss.beta=0.0 \
  training_schedule.losses.0.loss.diffclf_weight=1.0
```

**RNE training** (replace or combine with `diffclf_weight`):
```bash
  training_schedule.losses.0.loss.rne_weight=1.0 \
  training_schedule.losses.0.loss.rne_delta_t=1e-4
```

**DSM warmup** (score-only for the first N epochs, then auxiliary loss kicks in):
```bash
  training_schedule.dsm_warmup_epochs=500
```

**Time-dependent `log_Z` on the energy head:**
```bash
  +architecture.model.use_time_net=True
```

**MH-corrected reverse-time IID sampling at evaluation time:**
```bash
  evaluation.diffusion_with_mh=True \
  evaluation.diffusion_mh_steps=1 \
  evaluation.diffusion_num_steps=1000
```

**HMC MD step at evaluation time:**
```bash
  evaluation.with_mh=True
```

## Citation

If you use any of the additions above, please also cite:

```
@inproceedings{ouyang2026diffusiveclassificationlosslearning,
    title={A Diffusive Classification Loss for Learning Energy-based Generative Models},
    author={RuiKang OuYang and Louis Grenioux and José Miguel Hernández-Lobato},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://arxiv.org/abs/2601.21025},
}
```

# Contributing

Feel free to open an issue if you encounter any problems or have questions.

# Citation

If you find our work useful, please cite:

```
@inproceedings{plainer2025consistent,
  author = {Plainer, Michael and Wu, Hao and Klein, Leon and G{\"u}nnemann, Stephan and No{\'e}, Frank},
  title = {Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {Belgrave, D. and Zhang, C. and Lin, H. and Pascanu, R. and Koniusz, P. and Ghassemi, M. and Chen, N.},
  pages = {24460--24505},
  publisher = {Curran Associates, Inc.},
  volume = {38},
  year = {2025},
}
```
