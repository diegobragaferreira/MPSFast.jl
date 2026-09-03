# u1_path_block.jl — Phase 2: U(1) sector layout on the full MPS chain (path + label).
#
# Phase 1 block-sparsifies only the label site.  Phase 2 assigns bond charges on
# every site and provides symmetry-preserving projection on interior path bonds.

"""Per-site bond charges for the full classification MPS."""
struct U1ChainLayout
    site_left_charges::Vector{Vector{Int}}
    site_phys_charges::Vector{Vector{Int}}
    site_right_charges::Vector{Vector{Int}}
    spec::U1Spec
end

"""Build charge maps for every site from current tensor shapes."""
function build_u1_chain_layout(
    spec::U1Spec,
    mps::AbstractVector{<:AbstractArray{<:Real, 3}},
    n_classes::Int,
    d_path::Int,
)
    Ml = length(mps)
    slc = Vector{Vector{Int}}(undef, Ml)
    spc = Vector{Vector{Int}}(undef, Ml)
    src = Vector{Vector{Int}}(undef, Ml)
    @inbounds for j in 1:Ml
        A = mps[j]
        Dl, d, Dr = size(A)
        slc[j], spc[j], src[j] = site_bond_charges(
            spec, j, Ml, d_path, n_classes, Dl, d, Dr)
    end
    return U1ChainLayout(slc, spc, src, spec)
end

"""
    project_u1_path_bond!(X, spec, chain, j, Ml, Dl, d, d2, Dr; label_layout, strength)

Project merged bond tensor `X` at bond `j` onto U(1)-allowed blocks.
Delegates to `project_u1_merged_bond!` at `j = Ml-1` when `label_layout` is set.
"""
function project_u1_path_bond!(
    X::AbstractArray{T, 4},
    spec::U1Spec,
    chain::U1ChainLayout,
    j::Int,
    Ml::Int,
    Dl::Int,
    d::Int,
    d2::Int,
    Dr::Int;
    label_layout::Union{Nothing, U1LabelLayout} = nothing,
    strength::Real = 1.0,
) where {T<:Real}
    spec.mode == none && return X
    if j == Ml - 1 && label_layout !== nothing
        return project_u1_merged_bond!(
            X, spec, label_layout, j, Ml, Dl, d, d2, Dr; strength = strength)
    end
    s = clamp(Float64(strength), 0.0, 1.0)
    lc = chain.site_left_charges[j]
    pc = chain.site_phys_charges[j]
    pc2 = chain.site_phys_charges[j + 1]
    rc = chain.site_right_charges[j + 1]
    @inbounds for α in 1:Dl, σ in 1:d, τ in 1:d2, β in 1:Dr
        allowed = lc[α] + pc[σ] + pc2[τ] == rc[β]
        allowed || (X[α, σ, τ, β] *= T(1 - s))
    end
    return X
end

"""Fraction of merged-bond blocks allowed by flux rules at bond `j`."""
function u1_path_bond_allowed_fraction(
    chain::U1ChainLayout,
    j::Int,
    Ml::Int,
    Dl::Int,
    d::Int,
    d2::Int,
    Dr::Int;
    label_layout::Union{Nothing, U1LabelLayout} = nothing,
)
    if j == Ml - 1 && label_layout !== nothing
        return count(label_layout.allowed) / length(label_layout.allowed)
    end
    lc = chain.site_left_charges[j]
    pc = chain.site_phys_charges[j]
    pc2 = chain.site_phys_charges[j + 1]
    rc = chain.site_right_charges[j + 1]
    allowed = 0
    total = Dl * d * d2 * Dr
    @inbounds for α in 1:Dl, σ in 1:d, τ in 1:d2, β in 1:Dr
        lc[α] + pc[σ] + pc2[τ] == rc[β] && (allowed += 1)
    end
    return total > 0 ? allowed / total : 1.0
end

"""`true` when Phase-2 full-chain block mode is active for this spec."""
function u1_use_path_block_mode(spec::U1Spec, path_block_mode::Bool)
    path_block_mode && symmetry_active(spec)
end
