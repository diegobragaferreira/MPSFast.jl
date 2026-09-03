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
    train_ttn!, sample_ttn,
    init_ttn_classification, class_probabilities_ttn, predict_class_ttn,
    classification_accuracy_ttn,
    TTNInternalCut, ttn_subtree_leaves, ttn_internal_cuts, ttn_layer_entropy_summary

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
include("symmetry.jl")
include("su2_symmetry.jl")
include("u1_block_sparse.jl")
include("u1_path_block.jl")
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
export init_mps_classification

# ── U(1) symmetry (Phase 0: dense masking; Phase 1: block-sparse label) ───────
export SymmetryMode, U1Spec, NoSymmetry, symmetry_active
export u1_label_spec, u1_path_centered_spec, u1_path_data_matched_spec, u1_path_empirical_spec
export u1_path_empirical_from_probe, u1_path_sign_increment_spec, u1_path_upcount_spec
export apply_u1_conservation!, randomize_u1_allowed!, init_mps_classification_u1
export path_total_charge, flux_compatible, sector_compatibility_rate, flux_nearest_class
export flux_exact_class, bounded_q_label, empirical_q_labels, path_charges, empirical_path_charge_range
export mean_path_charge_by_class, u1_allowed_fraction, u1_mask_per_bond
export u1_forbidden_mass, u1_max_forbidden_entry, u1_symmetry_residual
export u1_training_step_audit
# ── SU(2) symmetry (TrigEncoder feature legs) ───────────────────────────────────
export SU2Mode, SU2Spec, NoSU2Symmetry, su2_active
export trig_feature_mz_charge, trig_feature_j_quantum
export su2_trig_feature_spec, su2_trig_bucket_path_spec, su2_as_u1_bucket_spec
export apply_su2_conservation!, randomize_su2_allowed!, init_mps_classification_su2
export su2_allowed_fraction, su2_symmetry_residual, trig_phi_weighted_charge
export su2_trig_empirical_bucket_spec, su2_bucket_compat
export su2_mask_per_bond
# ── U(1) Phase 1: block-sparse label site ─────────────────────────────────────
export U1LabelLayout, U1BlockLabel
export sector_aligned_bond_charges, label_interface_charges, build_u1_label_layout, allowed_label_pairs
export extract_u1_block_label, materialize_u1_label!, sync_u1_block_label!
export project_u1_label_site!, project_u1_merged_bond!, init_mps_classification_u1_block
export u1_block_label_stats, u1_block_forbidden_mass, u1_block_symmetry_residual
export mps_amplitude_u1_block, u1_block_training_step_audit, u1_use_block_mode
export safeguard_u1_label_norm!
export U1TrainingPlan, u1_training_plan
export u1_conservation_at_epoch, u1_data_compatible, site_bond_charges
# ── U(1) Phase 2: full-chain block layout (experimental) ───────────────────────
export U1ChainLayout, build_u1_chain_layout, project_u1_path_bond!
export u1_path_bond_allowed_fraction, u1_use_path_block_mode
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
export init_ttn_classification
export class_probabilities_ttn, predict_class_ttn, classification_accuracy_ttn
export TTNInternalCut, ttn_subtree_leaves, ttn_internal_cuts, ttn_layer_entropy_summary

end # module MPSFast
