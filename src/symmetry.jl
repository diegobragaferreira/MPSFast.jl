# symmetry.jl — U(1) sector metadata and conservation masking (Phase 0: dense MPS).

"""Symmetry mode for the TN ansatz (not data symmetry)."""
@enum SymmetryMode none U1_label U1_path_accumulating

"""
    U1Spec

Charge rules for a classification MPS.  When `mode == none`, all blocks are
allowed and every symmetry routine is a no-op (identical to the dense baseline).

* `q_path(σ)` — charge contributed by physical index `σ` on path sites.
* `q_label(c)` — charge on label site physical index `c` (class index).
* `sites` — `:all` masks every site; `:label_only` (path mode) masks only the
  label tensor using `Q(path) + q_label(c) == 0`, leaving path sites dense.
* `label_left_range` — optional `(qmin, qmax)` for the label-site left bond in
  path-accumulating mode (empirical `Q` support); `nothing` = full theoretical range.
"""
struct U1Spec
    mode::SymmetryMode
    q_path::Function
    q_label::Function
    sites::Symbol
    label_left_range::Union{Nothing, NTuple{2, Int}}
end

U1Spec(mode, q_path, q_label) = U1Spec(mode, q_path, q_label, :all, nothing)
U1Spec(mode, q_path, q_label, sites::Symbol) = U1Spec(mode, q_path, q_label, sites, nothing)

"""No symmetry: dense unrestricted tensors."""
const NoSymmetry = U1Spec(none, σ -> 0, c -> 0, :all, nothing)

"""`true` when `spec` enables U(1) flux rules (not the dense baseline)."""
symmetry_active(spec::U1Spec) = spec.mode != none

"""Path bonds neutral; label site carries class charges `0, 1, …`."""
function u1_label_spec(n_classes::Int)
    U1Spec(U1_label, σ -> 0, c -> c - 1, :all)
end

"""
    u1_path_centered_spec(d_path, n_classes; sites=:label_only)

Cumulative path charge from centered buckets; labels in sectors `0, 1, …`.

Default `sites=:label_only` enforces flux closure only at the label tensor
(recommended for Phase 0 dense masking).  Use `sites=:all` for full-chain
masking (very aggressive; can collapse the MPS).
"""
function u1_path_centered_spec(
    d_path::Int, n_classes::Int;
    sites::Symbol = :label_only,
)
    center = (d_path + 1) ÷ 2
    U1Spec(U1_path_accumulating, σ -> σ - center, c -> c - 1, sites, nothing)
end

"""Map class-conditional mean path charge to a bounded label sector in `{-1,0,1}`."""
function bounded_q_label(mean_Q::Real)
    isfinite(mean_Q) || return 0
    typical = clamp(round(Int, mean_Q), -1, 1)
    return -typical
end

"""All integer path charges `Q(x)` in encoded data (path sites only)."""
function path_charges(xi_data::AbstractMatrix{<:Integer}, spec::U1Spec)
    n = size(xi_data, 1)
    Qs = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        row = xi_data[i, :]
        Qs[i] = path_total_charge(view(row, 1:length(row) - 1), spec)
    end
    return Qs
end

"""Empirical `(qmin, qmax)` for path charge on encoded rows (e.g. 5–95% index range)."""
function empirical_path_charge_range(
    xi_data::AbstractMatrix{<:Integer},
    spec::U1Spec;
    q_lo::Real = 0.05,
    q_hi::Real = 0.95,
)
    Qs = sort!(path_charges(xi_data, spec))
    n = length(Qs)
    n == 0 && return (0, 0)
    i_lo = max(1, min(n, round(Int, 1 + q_lo * (n - 1))))
    i_hi = max(1, min(n, round(Int, 1 + q_hi * (n - 1))))
    return (Qs[i_lo], Qs[i_hi])
end

"""
    u1_path_data_matched_spec(d_path, n_classes, xi_data; sites=:label_only)

Bounded label sectors `q_label(c) ∈ {-1,0,1}` from class-conditional mean `Q(x)`
(clamped, then negated for flux closure).  Avoids huge charges like `-round(mean)=6`.
"""
function u1_path_data_matched_spec(
    d_path::Int, n_classes::Int, xi_data::AbstractMatrix{<:Integer};
    sites::Symbol = :label_only,
)
    probe = u1_path_centered_spec(d_path, n_classes; sites = sites)
    mean_Q = mean_path_charge_by_class(xi_data, probe, n_classes)
    q_labels = [bounded_q_label(mean_Q[c]) for c in 1:n_classes]
    center = (d_path + 1) ÷ 2
    U1Spec(U1_path_accumulating, σ -> σ - center, c -> q_labels[c], sites, nothing)
end

"""
    u1_path_empirical_spec(d_path, n_classes, xi_data; sites=:label_only, q_lo=0.05, q_hi=0.95)

Like `u1_path_data_matched_spec`, plus **`label_left_range`** from empirical path-charge
quantiles so bond indices on the label site cover observed `Q(x)` (Phase 0 C4).
"""
function u1_path_empirical_spec(
    d_path::Int, n_classes::Int, xi_data::AbstractMatrix{<:Integer};
    sites::Symbol = :label_only,
    q_lo::Real = 0.05,
    q_hi::Real = 0.95,
)
    probe = u1_path_centered_spec(d_path, n_classes; sites = sites)
    mean_Q = mean_path_charge_by_class(xi_data, probe, n_classes)
    q_labels = [bounded_q_label(mean_Q[c]) for c in 1:n_classes]
    ql, qr = empirical_path_charge_range(xi_data, probe; q_lo = q_lo, q_hi = q_hi)
  # include closure targets −q_label(c) in the bond range
    for c in 1:n_classes
        t = -q_labels[c]
        ql, qr = (min(ql, t), max(qr, t))
    end
    center = (d_path + 1) ÷ 2
    U1Spec(U1_path_accumulating, σ -> σ - center, c -> q_labels[c], sites, (ql, qr))
end

"""Map class-conditional mean path charge to a label sector (flux closure target)."""
function empirical_q_labels(
    xi_data::AbstractMatrix{<:Integer},
    probe::U1Spec,
    n_classes::Int;
    mode::Symbol = :bounded,
)
    mean_Q = mean_path_charge_by_class(xi_data, probe, n_classes)
    if mode == :bounded
        return [bounded_q_label(mean_Q[c]) for c in 1:n_classes]
    elseif mode == :round
        return [begin
            m = mean_Q[c]
            isfinite(m) ? Int(-round(m)) : 0
        end for c in 1:n_classes]
    else
        error("empirical_q_labels mode must be :bounded or :round, got $mode")
    end
end

"""
    u1_path_empirical_from_probe(probe, xi_data, n_classes; ...)

C4 empirical spec using an arbitrary path-charge probe (e.g. sign-increment or up-count).

`q_label_mode`:
* `:bounded` — `q_label ∈ {-1,0,1}` (default; good for centered charges).
* `:round` — `q_label(c) = -round(mean Q | class c)` (better for one-sided counts).
"""
function u1_path_empirical_from_probe(
    probe::U1Spec,
    xi_data::AbstractMatrix{<:Integer},
    n_classes::Int;
    q_lo::Real = 0.05,
    q_hi::Real = 0.95,
    q_label_mode::Symbol = :bounded,
)
    q_labels = empirical_q_labels(xi_data, probe, n_classes; mode = q_label_mode)
    ql, qr = empirical_path_charge_range(xi_data, probe; q_lo = q_lo, q_hi = q_hi)
    for c in 1:n_classes
        t = -q_labels[c]
        ql, qr = (min(ql, t), max(qr, t))
    end
    labels = copy(q_labels)
    return U1Spec(
        probe.mode, probe.q_path, c -> labels[c], probe.sites, (ql, qr))
end

"""
    u1_path_sign_increment_spec(d_path, n_classes; sites=:label_only)

Semantic U(1) for `SignIncrementEncoder`: `σ=1,2,3` → charges `-1,0,+1`.
`Q(x)` is the net up-minus-down increment count.
"""
function u1_path_sign_increment_spec(
    d_path::Int, n_classes::Int;
    sites::Symbol = :label_only,
)
    @assert d_path == 3 "sign-increment spec requires d_path=3"
    U1Spec(U1_path_accumulating, σ -> σ - 2, c -> c - 1, sites, nothing)
end

"""
    u1_path_upcount_spec(d_path, n_classes; sites=:label_only)

Semantic U(1) for `UpDownIncrementEncoder`: `σ=1,2` → charges `0,1`.
`Q(x)` counts up-increments (down and flat → 0).
"""
function u1_path_upcount_spec(
    d_path::Int, n_classes::Int;
    sites::Symbol = :label_only,
)
    @assert d_path == 2 "up-count spec requires d_path=2"
    U1Spec(U1_path_accumulating, σ -> σ - 1, c -> c - 1, sites, nothing)
end

"""Class index whose label charge best closes flux with path charge `Q`."""
function flux_nearest_class(
    xi_path::AbstractVector{<:Integer},
    spec::U1Spec,
    n_classes::Int,
)
    Q = path_total_charge(xi_path, spec)
    best_c, best_d = 1, typemax(Int)
    for c in 1:n_classes
        d = abs(Q + spec.q_label(c))
        if d < best_d
            best_d = d
            best_c = c
        end
    end
    return best_c
end

"""Class index with exact flux closure, or `0` if no sector matches."""
function flux_exact_class(
    xi_path::AbstractVector{<:Integer},
    spec::U1Spec,
    n_classes::Int,
)
    Q = path_total_charge(xi_path, spec)
    @inbounds for c in 1:n_classes
        Q + spec.q_label(c) == 0 && return c
    end
    return 0
end

@inline function _mask_this_site(spec::U1Spec, j::Int, Ml::Int)
    spec.mode == none && return false
    spec.mode == U1_path_accumulating && spec.sites == :label_only && return j == Ml
    return true
end

# ─── Charge ranges ───────────────────────────────────────────────────────────

function _q_path_range(spec::U1Spec, d_path::Int)
    spec.mode == none && return (0, 0)
    qs = [spec.q_path(σ) for σ in 1:d_path]
    return (minimum(qs), maximum(qs))
end

function _left_charge_range(spec::U1Spec, site::Int, Ml::Int, d_path::Int)
    spec.mode == none && return (0, 0)
    if spec.mode == U1_label
        site < Ml && return (0, 0)
        # label site left bond: need charges −q_label(c)
        qs = [-(spec.q_label(c)) for c in 1:max(2, site)]  # n_classes unknown here
        return (minimum(qs), maximum(qs))
    end
    # path accumulating: charge after (site-1) path contributions
    n_path = min(site - 1, Ml - 1)
    qmin, qmax = _q_path_range(spec, d_path)
    return (n_path * qmin, n_path * qmax)
end

function _right_charge_range(spec::U1Spec, site::Int, Ml::Int, d_path::Int)
    spec.mode == none && return (0, 0)
    if spec.mode == U1_label
        site < Ml && return (0, 0)
        return (0, 0)  # right bond of label site is dim 1, charge 0
    end
    n_path = min(site, Ml - 1)
    qmin, qmax = _q_path_range(spec, d_path)
    return (n_path * qmin, n_path * qmax)
end

"""Assign an integer charge to each bond index `1:D`, linearly spanning `[qmin, qmax]`."""
function _bond_index_charges(D::Int, qmin::Int, qmax::Int)
    D == 0 && return Int[]
    if qmin > qmax
        qmin, qmax = qmax, qmin
    end
    D == 1 && return [round(Int, (qmin + qmax) / 2)]
    return [round(Int, qmin + (k - 1) * (qmax - qmin) / (D - 1)) for k in 1:D]
end

function _phys_charges_site(spec::U1Spec, site::Int, Ml::Int, d_site::Int)
    if spec.mode == none
        return zeros(Int, d_site)
    end
    site < Ml && return [spec.q_path(σ) for σ in 1:d_site]
    return [spec.q_label(c) for c in 1:d_site]
end

function _block_allowed(spec::U1Spec, q_α::Int, q_σ::Int, q_β::Int)
    spec.mode == none && return true
    return q_α + q_σ == q_β
end

function _site_bond_charges(
    spec::U1Spec, j::Int, Ml::Int, d_path::Int, n_classes::Int,
    Dl::Int, d::Int, Dr::Int,
)
    if spec.mode == U1_path_accumulating && j == Ml
        if spec.label_left_range !== nothing
            ql, qr = spec.label_left_range
        else
            n_path = Ml - 1
            qmin, qmax = _q_path_range(spec, d_path)
            ql, qr = (n_path * qmin, n_path * qmax)
        end
    else
        ql, qr = _left_charge_range(spec, j, Ml, d_path)
        if spec.mode == U1_label && j == Ml
            qs = unique(-spec.q_label(c) for c in 1:n_classes)
            ql, qr = (minimum(qs), maximum(qs))
        end
    end
    qrr_min, qrr_max = _right_charge_range(spec, j, Ml, d_path)
    lc = _bond_index_charges(Dl, ql, qr)
    rc = Dr == 1 ? [0] : _bond_index_charges(Dr, qrr_min, qrr_max)
    pc = _phys_charges_site(spec, j, Ml, d)
    return lc, pc, rc
end

# ─── Masking ─────────────────────────────────────────────────────────────────

"""
    apply_u1_conservation!(mps, spec; mode=:hard, strength=1.0, n_classes=2, d_path=0)

Enforce U(1) flux rules on a dense classification MPS.

* `mode=:hard` — zero forbidden blocks.
* `mode=:soft` — multiply forbidden blocks by `(1 - strength)`.
* `mode=:none` — no-op.

Bond charges are recomputed from current tensor sizes each call (Phase 0 dense mask).
"""
function apply_u1_conservation!(
    mps::AbstractVector{<:AbstractArray{T,3}},
    spec::U1Spec;
    mode::Symbol = :hard,
    strength::Real = 1.0,
    n_classes::Int = 2,
    d_path::Int = 0,
) where {T<:Real}
    (spec.mode == none || mode == :none) && return mps
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))

    @inbounds for j in 1:Ml
        _mask_this_site(spec, j, Ml) || continue
        A = mps[j]
        Dl, d, Dr = size(A)
        lc, pc, rc = _site_bond_charges(spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            allowed = _block_allowed(spec, lc[α], pc[σ], rc[β])
            if !allowed
                if mode == :hard
                    A[α, σ, β] = zero(T)
                elseif mode == :soft
                    A[α, σ, β] = T((1 - Float64(strength)) * Float64(A[α, σ, β]))
                else
                    error("apply_u1_conservation!: unknown mode = $mode")
                end
            end
        end
    end
    return mps
end

"""Randomise only U(1)-allowed tensor entries (zeros forbidden blocks)."""
function randomize_u1_allowed!(
    mps::AbstractVector{<:AbstractArray{T,3}},
    spec::U1Spec;
    n_classes::Int = 2,
    d_path::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
    scale::Real = 1.0,
) where {T<:Real}
    spec.mode == none && return mps
    apply_u1_conservation!(mps, spec; mode = :hard, n_classes = n_classes, d_path = d_path)
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))
    @inbounds for j in 1:Ml
        _mask_this_site(spec, j, Ml) || continue
        A = mps[j]
        Dl, d, Dr = size(A)
        lc, pc, rc = _site_bond_charges(spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            if _block_allowed(spec, lc[α], pc[σ], rc[β])
                A[α, σ, β] = T(scale * randn(rng))
            else
                A[α, σ, β] = zero(T)
            end
        end
    end
    return mps
end

# ─── Path / flux diagnostics ─────────────────────────────────────────────────

"""Total U(1) charge accumulated along encoded path sites (excludes label)."""
function path_total_charge(xi_path::AbstractVector{<:Integer}, spec::U1Spec)
    spec.mode == none && return 0
    return sum(spec.q_path(Int(xi_path[t])) for t in eachindex(xi_path))
end

"""Whether `Q(path) + q_label(class) == 0` under the spec (strict flux closure)."""
function flux_compatible(
    xi_path::AbstractVector{<:Integer},
    class_idx::Int,
    spec::U1Spec,
)
    spec.mode == none && return true
    Q = path_total_charge(xi_path, spec)
    return Q + spec.q_label(class_idx) == 0
end

"""Fraction of rows in `xi_data` whose true label is flux-compatible."""
function sector_compatibility_rate(
    xi_data::AbstractMatrix{<:Integer},
    spec::U1Spec,
)
    spec.mode == none && return 1.0
    Nd = size(xi_data, 1)
    ok = 0
    @inbounds for i in 1:Nd
        row = xi_data[i, :]
        flux_compatible(view(row, 1:length(row)-1), Int(row[end]), spec) && (ok += 1)
    end
    return ok / Nd
end

"""
    u1_conservation_at_epoch(conservation, epoch, soft_epochs)

Effective conservation mode for training epoch `epoch`.
`:soft_then_hard` uses `:soft` for `epoch ≤ soft_epochs`, then `:hard`.
"""
function u1_conservation_at_epoch(
    conservation::Symbol,
    epoch::Int,
    soft_epochs::Int,
)
    conservation == :soft_then_hard && return epoch <= soft_epochs ? :soft : :hard
    return conservation
end

"""`true` when `sector_compatibility_rate(xi_train, spec) ≥ threshold`."""
function u1_data_compatible(
    xi_train::AbstractMatrix{<:Integer},
    spec::U1Spec;
    threshold::Real = 0.5,
)
    spec.mode == none && return true
    return sector_compatibility_rate(xi_train, spec) >= threshold
end

"""Bond charge vectors for site `j` (public wrapper for layout builders)."""
function site_bond_charges(
    spec::U1Spec, j::Int, Ml::Int, d_path::Int, n_classes::Int,
    Dl::Int, d::Int, Dr::Int,
)
    return _site_bond_charges(spec, j, Ml, d_path, n_classes, Dl, d, Dr)
end

"""Per-class mean path charge `Q(x)` (diagnostic for conserved quantities)."""
function mean_path_charge_by_class(
    xi_data::AbstractMatrix{<:Integer},
    spec::U1Spec,
    n_classes::Int,
)
    sums = zeros(Float64, n_classes)
    counts = zeros(Int, n_classes)
    @inbounds for i in 1:size(xi_data, 1)
        row = xi_data[i, :]
        c = Int(row[end])
        Q = path_total_charge(view(row, 1:length(row)-1), spec)
        sums[c] += Q
        counts[c] += 1
    end
    return [counts[c] > 0 ? sums[c] / counts[c] : NaN for c in 1:n_classes]
end

function init_mps_classification_u1(
    Ml::Int, d_path::Int, n_classes::Int, D_max::Int, spec::U1Spec;
    T::Type{<:Real} = Float32,
    rng::AbstractRNG = Random.default_rng(),
)
    mps = init_mps_classification(Ml, d_path, n_classes, D_max; T = T, rng = rng)
    spec.mode == none && return mps
    randomize_u1_allowed!(mps, spec; n_classes = n_classes, d_path = d_path, rng = rng)
    left_canonicalize_mps!(mps)
    apply_u1_conservation!(mps, spec; mode = :hard, n_classes = n_classes, d_path = d_path)
    return mps
end

function _safeguard_mps_norm!(mps::Vector{Array{T,3}}; floor::Real = 1e-8) where {T<:Real}
    n = norm(mps[end])
    if n >= floor && isfinite(n)
        return mps
    end
    Ml = length(mps)
    mps[Ml] .= T.(1e-2 .* randn(size(mps[Ml])))
    left_canonicalize_mps!(mps)
    return mps
end

"""Per-bond hard masking is safe when path sites are neutral (`U1_label`) or fully masked.
For `U1_path_accumulating` + `sites=:label_only`, masking only the label tensor after
every bond update zeros almost all label weights (wide left-bond charge range); mask at
epoch end instead."""
@inline function u1_mask_per_bond(spec::U1Spec, conservation::Symbol)
    (spec.mode == none || conservation == :none) && return false
    spec.mode == U1_path_accumulating && spec.sites == :label_only && return false
    return true
end

function _apply_u1_if_needed!(
    mps, spec::U1Spec, conservation::Symbol, strength::Real,
    n_classes::Int, d_path::Int;
    per_bond::Bool = true,
)
    (spec.mode == none || conservation == :none) && return
    per_bond || return
    apply_u1_conservation!(
        mps, spec;
        mode = conservation, strength = strength,
        n_classes = n_classes, d_path = d_path,
    )
    return nothing
end

function _apply_u1_epoch_end!(
    mps, spec::U1Spec, conservation::Symbol, strength::Real,
    n_classes::Int, d_path::Int,
)
    (spec.mode == none || conservation == :none) && return
    apply_u1_conservation!(
        mps, spec;
        mode = conservation, strength = strength,
        n_classes = n_classes, d_path = d_path,
    )
    _safeguard_mps_norm!(mps)
    return nothing
end

function u1_allowed_fraction(
    mps::AbstractVector{<:AbstractArray{<:Real,3}},
    spec::U1Spec;
    n_classes::Int = 2,
    d_path::Int = 0,
)
    spec.mode == none && return 1.0
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))
    allowed = 0
    total = 0
    @inbounds for j in 1:Ml
        Dl, d, Dr = size(mps[j])
        if !_mask_this_site(spec, j, Ml)
            allowed += Dl * d * Dr
            total += Dl * d * Dr
            continue
        end
        lc, pc, rc = _site_bond_charges(spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            total += 1
            _block_allowed(spec, lc[α], pc[σ], rc[β]) && (allowed += 1)
        end
    end
    return total > 0 ? allowed / total : 1.0
end

"""
    u1_forbidden_mass(mps, spec; ...)

Squared Frobenius norm in **forbidden** blocks vs total (masked sites only).
Returns `(forbidden_norm, total_norm, mass_ratio)`.
"""
function u1_forbidden_mass(
    mps::AbstractVector{<:AbstractArray{<:Real,3}},
    spec::U1Spec;
    n_classes::Int = 2,
    d_path::Int = 0,
)
    spec.mode == none && return (0.0, 0.0, 0.0)
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))
    forbid_sq = 0.0
    total_sq = 0.0
    @inbounds for j in 1:Ml
        _mask_this_site(spec, j, Ml) || continue
        A = mps[j]
        Dl, d, Dr = size(A)
        lc, pc, rc = _site_bond_charges(spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            v = Float64(A[α, σ, β])^2
            total_sq += v
            _block_allowed(spec, lc[α], pc[σ], rc[β]) || (forbid_sq += v)
        end
    end
    total_norm = sqrt(total_sq)
    forbid_norm = sqrt(forbid_sq)
    ratio = total_sq > 0 ? forbid_sq / total_sq : 0.0
    return (forbid_norm, total_norm, ratio)
end

"""Maximum `|entry|` in forbidden blocks (0 if spec is `NoSymmetry`)."""
function u1_max_forbidden_entry(
    mps::AbstractVector{<:AbstractArray{<:Real,3}},
    spec::U1Spec;
    n_classes::Int = 2,
    d_path::Int = 0,
)
    spec.mode == none && return 0.0
    Ml = length(mps)
    d_path > 0 || (d_path = size(mps[1], 2))
    mx = 0.0
    @inbounds for j in 1:Ml
        _mask_this_site(spec, j, Ml) || continue
        A = mps[j]
        Dl, d, Dr = size(A)
        lc, pc, rc = _site_bond_charges(spec, j, Ml, d_path, n_classes, Dl, d, Dr)
        for α in 1:Dl, σ in 1:d, β in 1:Dr
            _block_allowed(spec, lc[α], pc[σ], rc[β]) && continue
            mx = max(mx, abs(Float64(A[α, σ, β])))
        end
    end
    return mx
end

"""
    u1_symmetry_residual(mps, spec; atol=1e-10, ...)

`true` when every masked-site entry is either in an allowed block or has `|value| ≤ atol`.
Use after `apply_u1_conservation!` with `:hard` to verify exact symmetry.
"""
function u1_symmetry_residual(
    mps::AbstractVector{<:AbstractArray{<:Real,3}},
    spec::U1Spec;
    atol::Real = 1e-10,
    n_classes::Int = 2,
    d_path::Int = 0,
)
    return u1_max_forbidden_entry(mps, spec; n_classes = n_classes, d_path = d_path) <= atol
end
