# analysis.jl — Bipartite entropies and bond-spectrum logging.

# ─── Bond spectrum log ────────────────────────────────────────────────────────

"""
    log_bond_spectrum!(bond_log, epoch, sweep, j, s_full, keep, d_phys)

Append a named-tuple record to `bond_log` with the truncated singular values
`s[1:keep]`, von-Neumann entropy in nats, and normalised entropy `vn/(j·log d)`.
"""
function log_bond_spectrum!(
    bond_log::Vector, epoch::Int, sweep::Symbol, j::Int,
    s_full::AbstractVector, keep::Int, d_phys::Int,
)
    s  = Float64.(s_full[1:keep])
    p  = s .^ 2
    sp = sum(p)
    vn = (!isfinite(sp) || sp <= 0) ? NaN :
         sum(-pk * log(pk) for pk in (p ./ sp) if pk > 1e-30)
    logd    = log(Float64(d_phys))
    vn_norm = (!isfinite(vn) || j < 1 || logd <= 0) ? NaN : vn / (Float64(j) * logd)
    push!(bond_log, (; epoch, sweep, bond = j, s = copy(s), vn, vn_norm))
    return nothing
end

# ─── Bipartite entropies ──────────────────────────────────────────────────────

"""
    bipartite_entropies(mps) -> (S_values, entropies_nats)

Compute Schmidt spectra and von-Neumann entropies at every bipartite cut via
a right-to-left SVD sweep (does **not** mutate `mps`).

* `S_values[b]`       — descending Schmidt singular values across cut `b`
                         (between sites `b` and `b+1`).
* `entropies_nats[b]` — `−∑_α p_α log p_α` with `p_α = S²_α / ∑ S²`.

Assumes the MPS is left-canonical on entry (true after `train_mps!`);
call `left_canonicalize_mps!` first if unsure.
"""
function bipartite_entropies(mps::AbstractVector{<:AbstractArray{<:Real,3}})
    M = length(mps)
    M >= 2 || return (Vector{Vector{Float64}}(), Float64[])
    A    = Array{Float64,3}[Float64.(t) for t in mps]
    Svals = Vector{Vector{Float64}}(undef, M - 1)
    entr  = fill(NaN, M - 1)
    @inbounds for b in (M-1):-1:1
        Dl, d, Dr = size(A[b+1])
        F  = svd(reshape(A[b+1], Dl, d * Dr); full = false)
        s  = F.S
        rk = length(s)
        Svals[b] = collect(s)
        z = sum(abs2, s)
        if isfinite(z) && z > 0
            entr[b] = sum(-(pp/z) * log(pp/z) for pp in abs2.(s) if pp/z > 1e-30)
        end
        A[b+1] = reshape(Matrix(F.Vt[1:rk, :]), rk, d, Dr)
        Db1, db, _ = size(A[b])
        US = F.U[:, 1:rk] .* reshape(s, 1, :)
        A[b] = reshape(reshape(A[b], Db1 * db, :) * US, Db1, db, rk)
    end
    return Svals, entr
end

# ─── Entropy history ──────────────────────────────────────────────────────────

"""
    entropy_history(bond_log, M, n_epochs; sweep = :last) -> Matrix{Float64}

Extract a `(M-1) × n_epochs` matrix of bipartite von-Neumann entropies (nats)
from a `bond_log` produced by `train_mps!`.

`sweep` controls which half-epoch value is used per bond:
* `:last`     (default) — last logged value in the epoch.
* `:forward`  — forward-sweep value only.
* `:backward` — backward-sweep value only.
"""
function entropy_history(
    bond_log::AbstractVector, M::Int, n_epochs::Int;
    sweep::Symbol = :last,
)
    H = fill(NaN, M - 1, n_epochs)
    @inbounds for entry in bond_log
        e = entry.epoch
        b = entry.bond
        (1 <= b <= M-1 && 1 <= e <= n_epochs) || continue
        if sweep === :last
            H[b, e] = entry.vn
        elseif entry.sweep === sweep
            H[b, e] = entry.vn
        end
    end
    return H
end
