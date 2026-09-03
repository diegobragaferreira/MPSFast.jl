# u1_block_sparse.jl — Phase 1: sector-aligned U(1) label site + symmetry-preserving updates.
#
# Dense path sites; label tensor uses explicit charge sectors on the left bond.
# Training projects merged-bond gradients at j = Ml-1 so Adam/SVD never write
# forbidden flux blocks (no epoch-end masking massacre).

"""Sector layout for the classification label site (and penultimate path bond)."""
struct U1LabelLayout
    left_charges::Vector{Int}       # label-site left bond, length Dl
    class_charges::Vector{Int}      # label physical indices, length n_classes
    allowed::BitMatrix              # Dl × n_classes — flux-allowed blocks
    path_left_charges::Vector{Int}  # penultimate site left bond
    interface_charges::Vector{Int}  # shared path-right / label-left charges
end

"""Compact storage for allowed label blocks only (`pairs[k] = (α, c)`, `vals[k]`)."""
struct U1BlockLabel{T<:Real}
    layout::U1LabelLayout
    pairs::Vector{NTuple{2, Int}}
    vals::Vector{T}
end

n_allowed(layout::U1LabelLayout) = count(layout.allowed)
n_allowed(store::U1BlockLabel) = length(store.vals)

"""Assign label-interface bond charges, prioritising flux-closure targets `-q_label(c)`."""
function label_interface_charges(
    D::Int, ql::Int, qr::Int, class_charges::AbstractVector{Int},
)
    D <= 0 && return Int[]
    targets = unique(-c for c in class_charges)
    isempty(targets) && return fill(0, D)
    ql_eff = min(ql, minimum(targets))
    qr_eff = max(qr, maximum(targets))
    if D >= length(targets)
        out = copy(collect(targets))
        if length(out) < D
            pad = sector_aligned_bond_charges(D - length(out), ql_eff, qr_eff)
            append!(out, pad)
        end
        return out[1:D]
    end
    length(targets) == 1 && return fill(targets[1], D)
    D == 1 && return [targets[round(Int, (length(targets) + 1) / 2)]]
    idxs = round.(Int, range(1, length(targets), length = D))
    return [targets[i] for i in idxs]
end

"""Assign bond indices to integer charges, spreading `D` slots across `ql:qr`."""
function sector_aligned_bond_charges(D::Int, ql::Int, qr::Int)
    D <= 0 && return Int[]
    if ql > qr
        ql, qr = qr, ql
    end
    charges = collect(ql:qr)
    n = length(charges)
    D == 1 && return [round(Int, (ql + qr) / 2)]
    n == 1 && return fill(charges[1], D)
    if D <= n
        idxs = round.(Int, range(1, n, length = D))
        return [charges[i] for i in idxs]
    end
    out = Vector{Int}(undef, D)
    base, rem = divrem(D, n)
    pos = 1
    @inbounds for (i, q) in enumerate(charges)
        cnt = base + (i <= rem ? 1 : 0)
        for _ in 1:cnt
            pos > D && break
            out[pos] = q
            pos += 1
        end
    end
    return out
end

function _label_charge_range(spec::U1Spec, n_classes::Int, d_path::Int, Ml::Int)
    if spec.mode == U1_path_accumulating && spec.label_left_range !== nothing
        return spec.label_left_range
    end
    if spec.mode == U1_label
        qs = unique(-spec.q_label(c) for c in 1:n_classes)
        return (minimum(qs), maximum(qs))
    end
    n_path = Ml - 1
    qmin, qmax = _q_path_range(spec, d_path)
    return (n_path * qmin, n_path * qmax)
end

"""
    build_u1_label_layout(spec, mps, n_classes, d_path) -> U1LabelLayout

Charge maps for the label site and the path–label interface (site `Ml-1` right bond).
Uses sector-aligned charges when `symmetry_active(spec)`.
"""
function build_u1_label_layout(
    spec::U1Spec,
    mps::AbstractVector{<:AbstractArray{<:Real, 3}},
    n_classes::Int,
    d_path::Int,
)
    Ml = length(mps)
    A_label = mps[Ml]
    Dl, n_c, Dr = size(A_label)
    @assert n_c == n_classes && Dr == 1

    ql, qr = _label_charge_range(spec, n_classes, d_path, Ml)
    pc = [_phys_charges_site(spec, Ml, Ml, n_classes)[c] for c in 1:n_classes]
    lc = label_interface_charges(Dl, ql, qr, pc)
    allowed = falses(Dl, n_c)
    @inbounds for α in 1:Dl, c in 1:n_c
        allowed[α, c] = _block_allowed(spec, lc[α], pc[c], 0)
    end

    A_path = mps[Ml - 1]
    Dl_p, _, Dr_p = size(A_path)
    interface = label_interface_charges(Dr_p, ql, qr, pc)
    @assert length(interface) == Dl "label left bond Dl=$Dl must match path right bond Dr=$Dr_p"
    ql_p, qr_p = _left_charge_range(spec, Ml - 1, Ml, d_path)
    lc_p = sector_aligned_bond_charges(Dl_p, ql_p, qr_p)

    return U1LabelLayout(lc, pc, allowed, lc_p, interface)
end

"""List of allowed `(α, c)` pairs on the label site."""
function allowed_label_pairs(layout::U1LabelLayout)
    pairs = NTuple{2, Int}[]
    Dl, n_c = size(layout.allowed)
    @inbounds for α in 1:Dl, c in 1:n_c
        layout.allowed[α, c] && push!(pairs, (α, c))
    end
    return pairs
end

"""Extract compact label blocks from a dense label tensor."""
function extract_u1_block_label(
    A::AbstractArray{T, 3},
    layout::U1LabelLayout,
) where {T<:Real}
    pairs = allowed_label_pairs(layout)
    vals = Vector{T}(undef, length(pairs))
    @inbounds for k in eachindex(pairs)
        α, c = pairs[k]
        vals[k] = A[α, c, 1]
    end
    return U1BlockLabel{T}(layout, pairs, vals)
end

"""Write compact label values into dense `mps[Ml]` (forbidden blocks set to zero)."""
function materialize_u1_label!(
    mps::AbstractVector{<:AbstractArray{T, 3}},
    store::U1BlockLabel{T},
) where {T<:Real}
    A = mps[end]
    A .= zero(T)
    @inbounds for k in eachindex(store.pairs)
        α, c = store.pairs[k]
        A[α, c, 1] = store.vals[k]
    end
    return mps
end

"""Sync compact store from the dense label tensor (allowed entries only)."""
function sync_u1_block_label!(store::U1BlockLabel{T}, mps::AbstractVector{<:AbstractArray{T, 3}}) where {T<:Real}
    A = mps[end]
    @inbounds for k in eachindex(store.pairs)
        α, c = store.pairs[k]
        store.vals[k] = A[α, c, 1]
    end
    return store
end

"""Project label tensor onto allowed sectors (`strength=1` hard, `<1` soft dampening)."""
function project_u1_label_site!(
    A::AbstractArray{T, 3},
    layout::U1LabelLayout;
    strength::Real = 1.0,
) where {T<:Real}
    s = clamp(Float64(strength), 0.0, 1.0)
    Dl, n_c, Dr = size(A)
    @inbounds for α in 1:Dl, c in 1:n_c, β in 1:Dr
        layout.allowed[α, c] || (A[α, c, β] *= T(1 - s))
    end
    return A
end

"""
    project_u1_merged_bond!(X, spec, layout, j, Ml, Dl, d, d2, Dr)

Zero forbidden blocks in merged tensor `B` (or `grad`) at bond `j = Ml-1`.
Flux rule: `q_α + q_path(σ) + q_label(c) == 0` on the path–label interface.
"""
function project_u1_merged_bond!(
    X::AbstractArray{T, 4},
    spec::U1Spec,
    layout::U1LabelLayout,
    j::Int,
    Ml::Int,
    Dl::Int,
    d::Int,
    d2::Int,
    Dr::Int;
    strength::Real = 1.0,
) where {T<:Real}
    (spec.mode == none || j != Ml - 1) && return X
    s = clamp(Float64(strength), 0.0, 1.0)
    @inbounds for α in 1:Dl, σ in 1:d, c in 1:d2, β in 1:Dr
        qα = layout.path_left_charges[α]
        qσ = spec.q_path(σ)
        qc = layout.class_charges[c]
        qα + qσ + qc == 0 || (X[α, σ, c, β] *= T(1 - s))
    end
    return X
end

"""Apply label-site projection after a bond update touching the label tensor."""
function project_u1_after_bond!(
    mps::AbstractVector{<:AbstractArray{T, 3}},
    spec::U1Spec,
    layout::U1LabelLayout,
    j::Int,
) where {T<:Real}
    symmetry_active(spec) || return mps
    Ml = length(mps)
    j == Ml - 1 && project_u1_label_site!(mps[Ml], layout)
    return mps
end

"""
    init_mps_classification_u1_block(Ml, d_path, n_classes, D_max, spec; kwargs...)

Sector-aligned U(1) classification MPS for Phase 1 block-sparse label training.
Path sites are dense; label left bond uses `sector_aligned_bond_charges`.
Returns `(mps, layout, block_label)`.
"""
function init_mps_classification_u1_block(
    Ml::Int, d_path::Int, n_classes::Int, D_max::Int, spec::U1Spec;
    T::Type{<:Real} = Float32,
    rng::AbstractRNG = Random.default_rng(),
    sector_align::Bool = true,
)
    if !sector_align && symmetry_active(spec)
        mps = init_mps_classification_u1(Ml, d_path, n_classes, D_max, spec; T = T, rng = rng)
        layout = build_u1_label_layout(spec, mps, n_classes, d_path)
        project_u1_label_site!(mps[end], layout)
        block = extract_u1_block_label(mps[end], layout)
        return (mps, layout, block)
    end

    mps = init_mps_classification(Ml, d_path, n_classes, D_max; T = T, rng = rng)
    spec.mode == none && return (mps, build_u1_label_layout(spec, mps, n_classes, d_path),
                                 extract_u1_block_label(mps[end], build_u1_label_layout(spec, mps, n_classes, d_path)))

    layout = build_u1_label_layout(spec, mps, n_classes, d_path)
    A = mps[end]
    A .= zero(T)
    @inbounds for (α, c) in allowed_label_pairs(layout)
        A[α, c, 1] = T(randn(rng))
    end
    left_canonicalize_mps!(mps)
    project_u1_label_site!(mps[end], layout)
    block = extract_u1_block_label(mps[end], layout)
    return (mps, layout, block)
end

"""Memory / sparsity stats for Phase 1 vs dense label tensor."""
function u1_block_label_stats(
    mps::AbstractVector{<:AbstractArray{<:Real, 3}},
    layout::U1LabelLayout,
    block::U1BlockLabel;
    eltype_bytes::Int = 4,
)
    Ml = length(mps)
    A = mps[Ml]
    Dl, n_c, Dr = size(A)
    dense_n = Dl * n_c * Dr
    allowed_n = n_allowed(layout)
    compact_n = length(block.vals)
    return (;
        dense_params = dense_n,
        allowed_params = allowed_n,
        compact_params = compact_n,
        allowed_fraction = dense_n > 0 ? allowed_n / dense_n : 1.0,
        compact_bytes = compact_n * eltype_bytes,
        dense_bytes = dense_n * eltype_bytes,
    )
end

"""
    u1_block_forbidden_mass(mps, layout) -> (forbid_norm, total_norm, ratio)

Forbidden mass on the label site only (should be 0 under block mode).
"""
function u1_block_forbidden_mass(
    mps::AbstractVector{<:AbstractArray{<:Real, 3}},
    layout::U1LabelLayout,
)
    A = mps[end]
    forbid_sq = 0.0
    total_sq = 0.0
    Dl, n_c, Dr = size(A)
    @inbounds for α in 1:Dl, c in 1:n_c, β in 1:Dr
        v = Float64(A[α, c, β])^2
        total_sq += v
        layout.allowed[α, c] || (forbid_sq += v)
    end
    total_norm = sqrt(total_sq)
    forbid_norm = sqrt(forbid_sq)
    ratio = total_sq > 0 ? forbid_sq / total_sq : 0.0
    return (forbid_norm, total_norm, ratio)
end

"""`true` when label-site forbidden mass is below `atol`."""
function u1_block_symmetry_residual(
    mps::AbstractVector{<:AbstractArray{<:Real, 3}},
    layout::U1LabelLayout;
    atol::Real = 1e-10,
)
    _, _, ratio = u1_block_forbidden_mass(mps, layout)
    return ratio <= atol
end

"""
    mps_amplitude_u1_block(mps, x, layout; phi = nothing)

Amplitude with a fast label-site contraction using the flux mask (dense path sites).
"""
function mps_amplitude_u1_block(
    mps::AbstractVector{<:AbstractArray{T, 3}},
    x::AbstractVector{<:Integer},
    layout::U1LabelLayout;
    phi::Union{Nothing, AbstractMatrix} = nothing,
) where {T<:Real}
    phi === nothing || return mps_amplitude(mps, x; phi = phi)
    Ml = length(mps)
    @inbounds v = mps[1][1, x[1], :]'
    @inbounds for j in 2:(Ml - 1)
        v = v * mps[j][:, x[j], :]
    end
    c = x[Ml]
    A = mps[Ml]
    s = zero(T)
    @inbounds for α in axes(A, 1)
        layout.allowed[α, c] || continue
        s += v[1, α] * A[α, c, 1]
    end
    return s
end

"""
    u1_block_training_step_audit(mps, xi_data, spec, layout, j; ...)

Like `u1_training_step_audit` but uses gradient projection (block mode) instead of masking.
"""
function u1_block_training_step_audit(
    mps0::Vector{Array{T, 3}},
    xi_data,
    spec::U1Spec,
    layout::U1LabelLayout,
    j::Int;
    η::Real,
    D_max::Int,
    ε_cut::Real = 1e-5,
    n_classes::Int = 2,
    d_path::Int = 0,
    sweep::Symbol = :forward,
) where {T<:Real}
    mps = deepcopy(mps0)
    Nd = size(xi_data, 1)
    d_loc = size(mps[1], 2)
    d_path > 0 || (d_path = d_loc)
    Ml = length(mps)
    ws = TrainWorkspace(T, Nd, d_loc, D_max)
    _ensure_bins!(ws, xi_data)
    right_canonicalize_mps!(mps)
    Lenv, Renv = _canonical_envs_identity(mps)
    Lv = ones(T, Nd, 1)
    for k in 1:(j - 1)
        Lv = extend_lv_after_bond!(ws, mps, xi_data, k, Lv)
    end
    adam = AdamDict{T}()
    _, _, before_ratio = u1_block_forbidden_mass(mps, layout)
    update_pair!(ws, mps, xi_data, j, T(η), D_max, T(ε_cut), Lenv, Renv, adam;
                 Lv_carry = Lv, epoch = 1, sweep = sweep, d_phys = d_loc,
                 u1_layout = layout, u1_spec_proj = spec)
    layout_post = build_u1_label_layout(spec, mps, n_classes, d_path)
    _, _, after_update_ratio = u1_block_forbidden_mass(mps, layout_post)
    project_u1_label_site!(mps[end], layout_post)
    _, _, after_project_ratio = u1_block_forbidden_mass(mps, layout_post)
    residual_ok = u1_block_symmetry_residual(mps, layout_post)
    return (;
        bond = j, sweep,
        before_ratio,
        after_update_ratio,
        after_update_max = after_update_ratio > 0 ? sqrt(after_update_ratio) : 0.0,
        after_mask_ratio = after_project_ratio,
        residual_ok,
        block_mode = true,
    )
end

"""Re-init allowed label blocks when projection zeros the label norm."""
function safeguard_u1_label_norm!(
    mps::AbstractVector{<:AbstractArray{T, 3}},
    layout::U1LabelLayout;
    floor::Real = 1e-8,
    rng::AbstractRNG = Random.default_rng(),
    scale::Real = 1e-2,
) where {T<:Real}
    A = mps[end]
    n = norm(A)
    if n >= floor && isfinite(n)
        return mps
    end
    A .= zero(T)
    @inbounds for (α, c) in allowed_label_pairs(layout)
        A[α, c, 1] = T(scale * randn(rng))
    end
    return mps
end

"""Whether block mode should replace Phase 0 dense masking for this spec."""
function u1_use_block_mode(spec::U1Spec, block_mode::Bool)
    block_mode && symmetry_active(spec) && spec.sites == :label_only
end

"""Recommended U(1) training settings from data–spec compatibility."""
struct U1TrainingPlan
    use_symmetry::Bool
    use_block::Bool
    conservation::Symbol
    D_max_train::Int
    compat_rate::Float64
    reason::String
end

function u1_training_plan(
    xi_train::AbstractMatrix{<:Integer},
    spec::U1Spec;
    compat_threshold::Real = 0.5,
    prefer_block::Bool = true,
    D_max_dense::Int = 32,
    D_max_block::Int = 64,
    conservation::Symbol = :hard,
)
    compat = sector_compatibility_rate(xi_train, spec)
    if spec.mode == none || compat < compat_threshold
        reason = spec.mode == none ? "NoSymmetry" :
            "compat=$(round(compat; digits=3)) < threshold=$(compat_threshold)"
        return U1TrainingPlan(false, false, :none, D_max_dense, compat, reason)
    end
    use_block = prefer_block && u1_use_block_mode(spec, true)
    D = use_block ? D_max_block : D_max_dense
    reason = use_block ? "P1 block D=$D" : "P0 dense D=$D"
    return U1TrainingPlan(true, use_block, conservation, D, compat, reason)
end
