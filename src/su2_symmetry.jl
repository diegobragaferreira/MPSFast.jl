# su2_symmetry.jl — SU(2) spin-coupling masks for TrigEncoder feature legs (Phase 0 dense).

"""SU(2) symmetry mode (TN ansatz, not data symmetry)."""
@enum SU2Mode none SU2_trig_feature SU2_trig_bucket

"""
    SU2Spec

Spin-coupling rules for a classification MPS with `TrigEncoder` feature legs.

* `q_path(σ)` — twice the magnetic quantum number `m_z` on feature index `σ`
  (cos `hθ` → `+h`, sin `hθ` → `−h`).
* `j_path(σ)` — twice the spin `j` on feature `σ` (`j = h/2` for harmonic `h`).
* `q_label(c)`, `j_label(c)` — label-site quantum numbers.
* `sites` — `:path` (path tensors only), `:label_only`, or `:all`.
* `label_left_range` — optional `(mmin, mmax)` for label left bond (bucket mode).
"""
struct SU2Spec
    mode::SU2Mode
    q_path::Function
    j_path::Function
    q_label::Function
    j_label::Function
    sites::Symbol
    label_left_range::Union{Nothing, NTuple{2, Int}}
end

SU2Spec(mode, q_path, j_path, q_label, j_label) =
    SU2Spec(mode, q_path, j_path, q_label, j_label, :path, nothing)

const NoSU2Symmetry = SU2Spec(
    none, σ -> 0, σ -> 0, c -> 0, c -> 0, :path, nothing)

su2_active(spec::SU2Spec) = spec.mode != none

"""Twice-`m_z` charge for TrigEncoder feature index `σ` (harmonic pairs)."""
function trig_feature_mz_charge(σ::Int)
    h = (σ + 1) ÷ 2
    return isodd(σ) ? h : -h
end

"""Twice-spin `2j` for TrigEncoder feature index `σ`."""
function trig_feature_j_quantum(σ::Int)
    return (σ + 1) ÷ 2
end

"""
    su2_trig_feature_spec(d_features, n_classes; sites=:path)

SU(2) on trig **feature** legs: each harmonic pair is a spin-`j=h/2` doublet.
Bond blocks satisfy `m_α + m_σ = m_β` and `|J_α − J_σ| ≤ J_β ≤ J_α + J_σ`
(with `J = 2j` integers).
"""
function su2_trig_feature_spec(
    d_features::Int, n_classes::Int;
    sites::Symbol = :path,
)
    @assert iseven(d_features) && d_features >= 2
    n_harm = d_features ÷ 2
    @assert n_harm * 2 == d_features
    SU2Spec(
        SU2_trig_feature,
        trig_feature_mz_charge,
        trig_feature_j_quantum,
        c -> c - 1,
        c -> 0,
        sites,
        nothing,
    )
end

"""
    su2_trig_bucket_path_spec(K, n_classes; sites=:label_only)

Bucket-index path charge (for flux compat diagnostics on encoded `xi`).
Uses centered bucket index `k − (K+1)÷2` as an effective torus phase proxy.
Tensor masking reuses U(1) bucket charges via `su2_as_u1_bucket_spec`.
"""
function su2_trig_bucket_path_spec(
    K::Int, n_classes::Int;
    sites::Symbol = :label_only,
)
    center = (K + 1) ÷ 2
    SU2Spec(
        SU2_trig_bucket,
        k -> k - center,
        k -> 0,
        c -> c - 1,
        c -> 0,
        sites,
        nothing,
    )
end

"""Convert bucket-mode `SU2Spec` to a `U1Spec` for compat / masking on bucket indices."""
function su2_as_u1_bucket_spec(spec::SU2Spec, K::Int, n_classes::Int)
    spec.mode != SU2_trig_bucket &&
        error("su2_as_u1_bucket_spec requires SU2_trig_bucket mode")
    return U1Spec(
        U1_path_accumulating,
        spec.q_path,
        spec.q_label,
        spec.sites,
        spec.label_left_range,
    )
end

# ─── Bond (J, m) assignment ───────────────────────────────────────────────────

function _su2_j_range(spec::SU2Spec, d_site::Int)
    spec.mode == none && return (0, 0)
    js = [spec.j_path(σ) for σ in 1:d_site]
    return (minimum(js), maximum(js))
end

function _su2_m_range(spec::SU2Spec, d_site::Int)
    spec.mode == none && return (0, 0)
    qs = [spec.q_path(σ) for σ in 1:d_site]
    return (minimum(qs), maximum(qs))
end

function _su2_left_jm_range(spec::SU2Spec, site::Int, Ml::Int, d_path::Int)
    spec.mode == none && return (0, 0), (0, 0)
    n_path = min(site - 1, Ml - 1)
    jmin, jmax = _su2_j_range(spec, d_path)
    mmin, mmax = _su2_m_range(spec, d_path)
    return (n_path * jmin, n_path * jmax), (n_path * mmin, n_path * mmax)
end

function _su2_right_jm_range(spec::SU2Spec, site::Int, Ml::Int, d_path::Int)
    spec.mode == none && return (0, 0), (0, 0)
    n_path = min(site, Ml - 1)
    jmin, jmax = _su2_j_range(spec, d_path)
    mmin, mmax = _su2_m_range(spec, d_path)
    return (n_path * jmin, n_path * jmax), (n_path * mmin, n_path * mmax)
end

"""Linearly spaced integer `J` or `m` across bond indices `1:D`."""
function _bond_index_spread(D::Int, vmin::Int, vmax::Int)
    D == 0 && return Int[]
    if vmin > vmax
        vmin, vmax = vmax, vmin
    end
    D == 1 && return [round(Int, (vmin + vmax) / 2)]
    return [round(Int, vmin + (k - 1) * (vmax - vmin) / (D - 1)) for k in 1:D]
end

function _su2_site_bond_jm(
    spec::SU2Spec, j::Int, Ml::Int, d_path::Int, n_classes::Int,
    Dl::Int, d::Int, Dr::Int,
)
    if spec.mode == SU2_trig_bucket && j == Ml && spec.label_left_range !== nothing
        mmin, mmax = spec.label_left_range
        jmin, jmax = (0, 0)
    elseif spec.mode == SU2_trig_bucket && j == Ml
        n_path = Ml - 1
        mmin, mmax = _su2_m_range(spec, d_path)
        mmin, mmax = (n_path * mmin, n_path * mmax)
        jmin, jmax = (0, 0)
    else
        (jmin, jmax), (mmin, mmax) = _su2_left_jm_range(spec, j, Ml, d_path)
    end
    (jrmin, jrmax), (mrmin, mrmax) = _su2_right_jm_range(spec, j, Ml, d_path)

    Jl = _bond_index_spread(Dl, jmin, jmax)
    Jr = Dr == 1 ? [0] : _bond_index_spread(Dr, jrmin, jrmax)
    ml = _bond_index_spread(Dl, mmin, mmax)
    mr = Dr == 1 ? [0] : _bond_index_spread(Dr, mrmin, mrmax)

    Jσ = [spec.j_path(σ) for σ in 1:d]
    mσ = [spec.q_path(σ) for σ in 1:d]
  # label site physical indices
    if j == Ml
        Jσ = [spec.j_label(c) for c in 1:d]
        mσ = [spec.q_label(c) for c in 1:d]
    end
    return Jl, ml, Jσ, mσ, Jr, mr
end

@inline function _su2_block_allowed(
    Jα::Int, mα::Int, Jσ::Int, mσ::Int, Jβ::Int, mβ::Int,
)
    mα + mσ == mβ || return false
    abs(mβ) > Jβ && return false
    Jβ < abs(Jα - Jσ) && return false
    Jβ > Jα + Jσ && return false
    return true
end

@inline function _mask_su2_site(spec::SU2Spec, j::Int, Ml::Int)
    spec.mode == none && return false
    spec.sites == :all && return true
    spec.sites == :path && return j < Ml
    spec.sites == :label_only && return j == Ml
    return true
end

"""
    apply_su2_conservation!(mps, spec; mode=:hard, strength=1.0, n_classes=2, d_path=0)

Enforce SU(2) spin-coupling rules on dense classification MPS tensors.
"""
function apply_su2_conservation!(
    mps::AbstractVector{<:AbstractArray{T,3}},
    spec::SU2Spec;
    mode::Symbol = :hard,
    strength::Real = 1.0,
    n_classes::Int = 2,
    d_path::Int = 0,
) where {T<:Real}
    (spec.mode == none || mode == :none) && return mps
    spec.mode == SU2_trig_bucket && return mps  # bucket mode uses U1 compat only
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))

    @inbounds for j in 1:Ml
        _mask_su2_site(spec, j, Ml) || continue
        A = mps[j]
        Dl, d, Dr = size(A)
        Jl, ml, Jσ, mσ, Jr, mr = _su2_site_bond_jm(
            spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            allowed = _su2_block_allowed(Jl[α], ml[α], Jσ[σ], mσ[σ], Jr[β], mr[β])
            if !allowed
                if mode == :hard
                    A[α, σ, β] = zero(T)
                elseif mode == :soft
                    A[α, σ, β] = T((1 - Float64(strength)) * Float64(A[α, σ, β]))
                else
                    error("apply_su2_conservation!: unknown mode = $mode")
                end
            end
        end
    end
    return mps
end

function randomize_su2_allowed!(
    mps::AbstractVector{<:AbstractArray{T,3}},
    spec::SU2Spec;
    n_classes::Int = 2,
    d_path::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    scale::Real = 1.0,
) where {T<:Real}
    spec.mode == none && return mps
    spec.mode == SU2_trig_bucket && return mps
    apply_su2_conservation!(mps, spec; mode = :hard, n_classes = n_classes, d_path = d_path)
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))
    @inbounds for j in 1:Ml
        _mask_su2_site(spec, j, Ml) || continue
        A = mps[j]
        Dl, d, Dr = size(A)
        Jl, ml, Jσ, mσ, Jr, mr = _su2_site_bond_jm(
            spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            if _su2_block_allowed(Jl[α], ml[α], Jσ[σ], mσ[σ], Jr[β], mr[β])
                A[α, σ, β] = T(scale * randn(rng))
            else
                A[α, σ, β] = zero(T)
            end
        end
    end
    return mps
end

function init_mps_classification_su2(
    Ml::Int, d_path::Int, n_classes::Int, D_max::Int, spec::SU2Spec;
    K::Int = 0,
    T::Type{<:Real} = Float32,
    rng::AbstractRNG = Random.default_rng(),
)
    mps = init_mps_classification(Ml, d_path, n_classes, D_max; T = T, rng = rng)
    spec.mode == none && return mps
    if spec.mode == SU2_trig_bucket
        K > 0 || error("init_mps_classification_su2: bucket mode requires K > 0")
        u1 = su2_as_u1_bucket_spec(spec, K, n_classes)
        randomize_u1_allowed!(mps, u1; n_classes = n_classes, d_path = d_path, rng = rng)
        left_canonicalize_mps!(mps)
        apply_u1_conservation!(mps, u1; mode = :hard, n_classes = n_classes, d_path = d_path)
    else
        randomize_su2_allowed!(mps, spec; n_classes = n_classes, d_path = d_path, rng = rng)
        left_canonicalize_mps!(mps)
        apply_su2_conservation!(mps, spec; mode = :hard, n_classes = n_classes, d_path = d_path)
    end
    return mps
end

function su2_allowed_fraction(
    mps::AbstractVector{<:AbstractArray{<:Real,3}},
    spec::SU2Spec;
    n_classes::Int = 2,
    d_path::Int = 0,
)
    spec.mode == none && return 1.0
    if spec.mode == SU2_trig_bucket
        return 1.0  # not applied on feature legs
    end
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))
    allowed = 0
    total = 0
    @inbounds for j in 1:Ml
        Dl, d, Dr = size(mps[j])
        if !_mask_su2_site(spec, j, Ml)
            allowed += Dl * d * Dr
            total += Dl * d * Dr
            continue
        end
        Jl, ml, Jσ, mσ, Jr, mr = _su2_site_bond_jm(
            spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            total += 1
            _su2_block_allowed(Jl[α], ml[α], Jσ[σ], mσ[σ], Jr[β], mr[β]) && (allowed += 1)
        end
    end
    return total > 0 ? allowed / total : 1.0
end

"""Max |tensor entry| in SU(2)-forbidden blocks (should be ≈0 after hard projection)."""
function su2_symmetry_residual(
    mps::AbstractVector{<:AbstractArray{<:Real,3}},
    spec::SU2Spec;
    n_classes::Int = 2,
    d_path::Int = 0,
)
    spec.mode == none && return 0.0
    spec.mode == SU2_trig_bucket && return 0.0
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))
    res = 0.0
    @inbounds for j in 1:Ml
        _mask_su2_site(spec, j, Ml) || continue
        A = mps[j]
        Dl, d, Dr = size(A)
        Jl, ml, Jσ, mσ, Jr, mr = _su2_site_bond_jm(
            spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            _su2_block_allowed(Jl[α], ml[α], Jσ[σ], mσ[σ], Jr[β], mr[β]) && continue
            res = max(res, abs(Float64(A[α, σ, β])))
        end
    end
    return res
end

"""
    trig_phi_weighted_charge(k, phi) -> Float64

Effective `m_z` charge of bucket row `k` under feature map `phi` (diagnostic).
"""
function trig_phi_weighted_charge(k::Int, phi::AbstractMatrix{<:Real})
    row = @view phi[k, :]
    d = length(row)
    wsum = sum(abs, row)
    wsum == 0 && return 0.0
    q = 0.0
    @inbounds for σ in 1:d
        q += abs(row[σ]) * trig_feature_mz_charge(σ)
    end
    return q / wsum
end

"""
    su2_trig_empirical_bucket_spec(K, n_classes, xi_data; ...)

Empirical bucket-flux spec (C4-style) for TrigEncoder encoded data.
"""
function su2_trig_empirical_bucket_spec(
    K::Int,
    n_classes::Int,
    xi_data::AbstractMatrix{<:Integer};
    q_lo::Real = 0.05,
    q_hi::Real = 0.95,
    q_label_mode::Symbol = :bounded,
)
    probe = su2_trig_bucket_path_spec(K, n_classes)
    u1_probe = su2_as_u1_bucket_spec(probe, K, n_classes)
    q_labels = empirical_q_labels(xi_data, u1_probe, n_classes; mode = q_label_mode)
    ql, qr = empirical_path_charge_range(xi_data, u1_probe; q_lo = q_lo, q_hi = q_hi)
    for c in 1:n_classes
        t = -q_labels[c]
        ql, qr = (min(ql, t), max(qr, t))
    end
    return SU2Spec(
        SU2_trig_bucket,
        probe.q_path,
        probe.j_path,
        c -> q_labels[c],
        probe.j_label,
        probe.sites,
        (ql, qr),
    )
end

function su2_bucket_compat(
    xi_data::AbstractMatrix{<:Integer},
    spec::SU2Spec,
    K::Int,
    n_classes::Int,
)
    spec.mode != SU2_trig_bucket && return 1.0
    u1 = su2_as_u1_bucket_spec(spec, K, n_classes)
    return sector_compatibility_rate(xi_data, u1)
end

@inline function su2_mask_per_bond(spec::SU2Spec, conservation::Symbol)
    (spec.mode == none || conservation == :none) && return false
    spec.mode == SU2_trig_bucket && return false
    return spec.sites == :all
end

function _apply_su2_epoch_end!(
    mps, spec::SU2Spec, conservation::Symbol, strength::Real,
    n_classes::Int, d_path::Int,
)
    (spec.mode == none || conservation == :none) && return
    apply_su2_conservation!(
        mps, spec;
        mode = conservation, strength = strength,
        n_classes = n_classes, d_path = d_path,
    )
    _safeguard_mps_norm!(mps)
    return nothing
end
