# MPSFast.jl

Fast MPS (Matrix Product State) Born-machine training for discrete path distributions, with applications to options pricing and classification on time series.

Based on the methods of [Kobayashi, Suimon & Miyamoto (2024)](https://arxiv.org/abs/2402.17148) — *Time series generation for option pricing on quantum computers using tensor networks*.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/diegobragaferreira/MPSFast.jl")
```

Or, for local development:

```julia
Pkg.develop(path = "/path/to/MPSFast.jl")
```

## Quick start

```julia
using MPSFast
using MPSFast.Encoders
using Random

# 1. Simulate some paths (e.g. Heston or GBM)
rng  = MersenneTwister(42)
N, M = 2000, 20                   # samples × timesteps
paths = cumsum(randn(rng, N, M) .* 0.02, dims = 2) .+ 1.0

# 2. Encode
enc = BasisEncoder(4)             # d = 2^4 = 16 buckets per timestep
fit_grid!(enc, paths)
xi  = encode_paths(enc, paths)    # (N, M) Int matrix

# 3. Initialise MPS and train
mps = init_mps(size(xi, 2), site_dim(enc), 16)
nll_hist = train_mps!(mps, xi, 10, 1e-3, 16, 1e-5)

# 4. Sample new paths
sampled_paths, _ = sample_paths(enc, mps, 500)
```

See `notebooks/getting_started.ipynb` for a self-contained walkthrough.

## Package layout

```
src/
  MPSFast.jl      — module root, exports
  core.jl         — MPS amplitude, gauge, transfer matrices, norm environments
  training.jl     — DMRG-style training: workspace, NLL gradient, Adam, bond sweep
  sampling.jl     — sequential conditional sampling (feature-map aware)
  analysis.jl     — bipartite entropies, bond-spectrum logging
  io.jl           — JLD2 checkpointing (save / load bundles)
  Encoders.jl     — path encoders submodule (Basis / Binary / Trig)

test/
  runtests.jl     — unit and smoke tests

notebooks/
  getting_started.ipynb     — zero-to-trained MPS in one notebook
  dmrg_tutorial.ipynb       — step-by-step DMRG algorithm walkthrough
  experiments/
    paper_reproduction.ipynb      — reproduces Kobayashi 2024 options pricing
    encodings_comparison.ipynb    — Basis vs Binary vs Trig encoder study
    classification.ipynb          — supervised Born-machine classification
```

## Encoders

| Encoder | Chain length | Site dim `d` | Feature map `Φ` |
|---------|-------------|--------------|-----------------|
| `BasisEncoder(m)` | `M` | `2^m` | `nothing` (one-hot) |
| `BinaryEncoder(m)` | `M·m` | `2` | `nothing` (one-hot) |
| `TrigEncoder(m, d_feat)` | `M` | `d_feat` | `K×d_feat` trig harmonics |

`TrigEncoder` activates Gram-weighted transfer matrices (`G = Φ'Φ`) inside the training kernel — lower bond dimension at the same expressive capacity.

## Conventions

- MPS tensors: `A[j] :: Array{T,3}` with layout `(D_left, d, D_right)`.
- Physical indices are 1-based (`σ ∈ 1:d`).
- Training assumes `Float32` tensors by default (faster on Apple Silicon / CUDA).
- All heavy contractions reduce to a small number of BLAS calls (`LinearAlgebra.mul!` / `axpy!`).

## Performance tips

```bash
export JULIA_NUM_THREADS=auto    # use all physical cores
export OPENBLAS_NUM_THREADS=1    # avoid BLAS thread contention with Julia threads
```

On Apple Silicon, link against `AppleAccelerate` for further BLAS speedup:

```julia
using AppleAccelerate  # add before the first LinearAlgebra call
```

## Citation

```bibtex
@article{kobayashi2024timeseries,
  title  = {Time series generation for option pricing on quantum computers using tensor networks},
  author = {Kobayashi, Tsubasa and Suimon, Yusuke and Miyamoto, Koichi},
  year   = {2024},
  url    = {https://arxiv.org/abs/2402.17148}
}
```
