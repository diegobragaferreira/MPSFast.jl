"""
    MPSFast

Fast MPS and TTN Born-machine training for discrete path distributions.

Provides DMRG-style BLAS-batched training, truncated-SVD bond updates,
in-place Adam optimiser, sequential conditional sampling, bipartite
entropies, and JLD2 checkpointing.

Also provides a Binary Tree Tensor Network (BTT) Born machine (`BinaryTTN`)
with analogous training (`train_ttn!`) and sampling (`sample_ttn`).

The `Encoders` submodule translates continuous paths into integer physical-leg
indices and, optionally, a feature matrix Φ for Gram-weighted inner products.

# Public API (re-exported at package level)

    mps_amplitude, log_partition_function, left_canonicalize_mps!, right_canonicalize_mps!,
    norm_environments, refresh_norm_envs_after_bond!,
    class_probabilities, predict_class, classification_accuracy,
    init_mps,
    TrainWorkspace, train_mps!, nll_gradient!, update_pair!,
    boundary_vectors, lv_prefix, extend_lv_after_bond!,
    sample_paths_feature_map,
    bipartite_entropies, entropy_history, log_bond_spectrum!,
    save_mps_bundle, load_mps_bundle, load_bond_log,
    BinaryTTN, init_ttn, ttn_amplitude, ttn_nll, root_canonicalize_ttn!,
    train_ttn!, sample_ttn

# Submodule

    MPSFast.Encoders  — path encoders (BasisEncoder, BinaryEncoder, TrigEncoder)
"""
module MPSFast

using LinearAlgebra
using JLD2
using Random
using Base.Threads
using TSVD

include("core.jl")
include("training.jl")
include("sampling.jl")
include("analysis.jl")
include("io.jl")
include("Encoders.jl")
include("ttn.jl")

# ── Core ──────────────────────────────────────────────────────────────────────
export mps_amplitude
export log_partition_function
export left_canonicalize_mps!
export right_canonicalize_mps!
export norm_environments
export refresh_norm_envs_after_bond!
export class_probabilities, predict_class, classification_accuracy
export init_mps

# ── Training ──────────────────────────────────────────────────────────────────
export TrainWorkspace
export train_mps!, cosine_lr
export nll_gradient!
export update_pair!
export boundary_vectors
export lv_prefix
export extend_lv_after_bond!

# ── Sampling ──────────────────────────────────────────────────────────────────
export sample_paths_feature_map

# ── Analysis ──────────────────────────────────────────────────────────────────
export bipartite_entropies
export entropy_history
export log_bond_spectrum!

# ── I/O ───────────────────────────────────────────────────────────────────────
export save_mps_bundle
export load_mps_bundle
export load_bond_log

# ── TTN ───────────────────────────────────────────────────────────────────────
export BinaryTTN
export init_ttn
export ttn_amplitude
export ttn_nll
export root_canonicalize_ttn!
export train_ttn!
export sample_ttn

end # module MPSFast
