# market_making.jl — Ground-state tensor-network formulation of multi-asset
# market making (G. Coutinho, "Ground-State Tensor Networks for Multi-Asset
# Market Making", internal report, 28 July 2026).
#
# Milestone 1 (report §12, item 1, extended to items 2-3):
#   * exact sparse inventory Hamiltonian (report Appendix A, eq. 22)
#   * exact bond-(K+2) MPO for factor-structured covariance (Prop. 5.2, eq. 46)
#     with a dense-contraction cross-check against the sparse Hamiltonian
#   * sparse ground/first-excited state via Lanczos (KrylovKit)
#   * bipartite (Schmidt) entropies of the exact ground state (§8.2)
#   * sequential TT-SVD compression + state/energy/residual/quote-ratio error
#     metrics reproducing report Table 1 (§8.3)
#
# Symmetric-liquidity case only (ηᵇ = ηᵃ =: η), matching the report's own
# reproducible 8-asset example (§8.1) and Appendix A pseudocode.

using SparseArrays
using KrylovKit: eigsolve
using LinearAlgebra

# ── Model ─────────────────────────────────────────────────────────────────────

"""
    MarketMakingModel(N, Qs, Σ, μ, k, γ, η)

Symmetric-liquidity multi-asset inventory market-making model (report §2, §4).

* `N`  — number of assets
* `Qs` — inventory limits `Q_i` (local Hilbert space dimension `d_i = 2Q_i+1`)
* `Σ`  — instantaneous price covariance (`N×N`)
* `μ`  — drift (`N`)
* `k`  — common exponential fill-intensity slope
* `γ`  — CARA risk aversion
* `η`  — symmetric hopping coefficients `η_i` (eq. 12 with `ηᵇ_i = ηᵃ_i`)
"""
struct MarketMakingModel
    N::Int
    Qs::Vector{Int}
    Σ::Matrix{Float64}
    μ::Vector{Float64}
    k::Float64
    γ::Float64
    η::Vector{Float64}
end

"""Risk-potential coefficient `c = kγ/2` (report eq. 12)."""
c_coeff(m::MarketMakingModel) = m.k * m.γ / 2

"""Local Hilbert-space dimensions `d_i = 2Q_i + 1`."""
site_dims(m::MarketMakingModel) = 2 .* m.Qs .+ 1

"""Full inventory-grid size `|Q| = ∏(2Q_i+1)`."""
hilbert_dim(m::MarketMakingModel) = prod(site_dims(m))

# ── Indexing (lexicographic, site 1 slowest — matches report Appendix A) ──────

"""1-based linear index of inventory configuration `q` (site 1 slowest-varying)."""
function config_to_index(q::AbstractVector{<:Integer}, Qs::AbstractVector{<:Integer})
    N = length(Qs)
    idx = 0
    @inbounds for i in 1:N
        σi = q[i] + Qs[i]              # 0-based physical index
        idx = idx * (2Qs[i] + 1) + σi
    end
    return idx + 1
end

"""Inverse of [`config_to_index`](@ref): 1-based linear index → inventory vector."""
function index_to_config(idx1::Integer, Qs::AbstractVector{<:Integer})
    N = length(Qs)
    dims = 2 .* Qs .+ 1
    strides = ones(Int, N)
    @inbounds for i in (N - 1):-1:1
        strides[i] = strides[i + 1] * dims[i + 1]
    end
    idx = idx1 - 1
    q = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        σi = div(idx, strides[i])
        idx = idx % strides[i]
        q[i] = σi - Qs[i]
    end
    return q
end

# ── Local operators (report §4.1) ──────────────────────────────────────────────

"""
    local_shift_ops(Q) -> (Lp, Lm)

Truncated pull-shift operators on a single site of dimension `d = 2Q+1`
(report eq. 20-21): `Lp[σ,σ+1] = 1` for `σ=1:d-1` (lowers inventory by one
unit), `Lm = Lp'` (raises inventory by one unit).
"""
function local_shift_ops(Q::Int)
    d = 2Q + 1
    Lp = spzeros(Float64, d, d)
    @inbounds for σ in 1:(d - 1)
        Lp[σ, σ + 1] = 1.0
    end
    return Lp, sparse(Lp')
end

"""Embed a single-site `d×d` operator at position `i`, identity elsewhere."""
function kron_at(op::AbstractMatrix{<:Real}, i::Int, dims::Vector{Int})
    N = length(dims)
    out = i == 1 ? sparse(Float64.(op)) : sparse(1.0I, dims[1], dims[1])
    @inbounds for j in 2:N
        factor = j == i ? sparse(Float64.(op)) : sparse(1.0I, dims[j], dims[j])
        out = kron(out, factor)
    end
    return out
end

# ── Exact sparse Hamiltonian (report Appendix A) ───────────────────────────────

"""
    build_hamiltonian_sparse(model) -> SparseMatrixCSC{Float64,Int}

Assemble the exact inventory Hamiltonian (eq. 22) in lexicographic
tensor-product order, following the report's Appendix A pseudocode:
a diagonal quadratic-form potential `V(q) = c q'Σq - k μ'q` plus a
Kronecker-sum of local hopping terms `-η_i(L⁺_i + L⁻_i)`.
"""
function build_hamiltonian_sparse(model::MarketMakingModel)
    N, Qs, Σ, μ, k = model.N, model.Qs, model.Σ, model.μ, model.k
    c = c_coeff(model)
    dims = site_dims(model)
    D = prod(dims)

    V = Vector{Float64}(undef, D)
    @inbounds for lin in 1:D
        q = index_to_config(lin, Qs)
        V[lin] = c * dot(q, Σ * q) - k * dot(μ, q)
    end
    H = spdiagm(0 => V)

    @inbounds for i in 1:N
        Lp, Lm = local_shift_ops(Qs[i])
        Si = Lp + Lm
        H = H - model.η[i] .* kron_at(Si, i, dims)
    end
    return H
end

# ── Exact K+2 MPO for factor covariance Σ = Δ + BBᵀ (report §5, Prop. 5.2) ────

"""
    build_mpo_cores(model, Δ, B) -> Vector{Array{Float64,4}}

Exact matrix-product-operator representation of bond dimension `K+2`
(report eq. 46) for the factor-structured Hamiltonian with covariance
`Σ = Δ + BBᵀ`, `Δ` diagonal (`N`-vector), `B ∈ R^{N×K}`. Core `i` has shape
`(Rleft, d_i, d_i, Rright)`; boundary bonds have size 1.
"""
function build_mpo_cores(model::MarketMakingModel, Δ::Vector{Float64}, B::Matrix{Float64})
    N, Qs, μ, k = model.N, model.Qs, model.μ, model.k
    c = c_coeff(model)
    K = size(B, 2)
    R = K + 2
    dims = site_dims(model)
    cores = Vector{Array{Float64,4}}(undef, N)
    @inbounds for i in 1:N
        d = dims[i]
        Lp, Lm = local_shift_ops(Qs[i])
        Si = Matrix(Lp + Lm)
        qi = Diagonal(Float64.(-Qs[i]:Qs[i]))
        onsite_coeff = Δ[i] + sum(B[i, α]^2 for α in 1:K)
        hi = c * onsite_coeff .* Matrix(qi^2) .- k * μ[i] .* Matrix(qi) .- model.η[i] .* Si

        W = zeros(Float64, R, d, d, R)
        W[1, :, :, 1] .= Matrix(1.0I, d, d)
        W[R, :, :, R] .= Matrix(1.0I, d, d)
        W[1, :, :, R] .= hi
        for α in 1:K
            W[1, :, :, 1 + α] .= B[i, α] .* Matrix(qi)
            W[1 + α, :, :, 1 + α] .= Matrix(1.0I, d, d)
            W[1 + α, :, :, R] .= (2c * B[i, α]) .* Matrix(qi)
        end
        # Boundary bonds have size 1: site 1 exposes only row 1 (left boundary
        # selects the first auxiliary state), site N exposes only column R
        # (right boundary selects the last auxiliary state).
        if i == 1 && N == 1
            cores[i] = reshape(W[1, :, :, R], 1, d, d, 1)
        elseif i == 1
            cores[i] = reshape(W[1, :, :, :], 1, d, d, R)
        elseif i == N
            cores[i] = reshape(W[:, :, :, R], R, d, d, 1)
        else
            cores[i] = W
        end
    end
    return cores
end

"""
    mpo_to_dense(cores) -> Matrix{Float64}

Contract MPO cores `(Rleft,d,d,Rright)` into the dense `D×D` operator matrix.
Only intended for cross-checking small systems (report §12, milestone 1).
"""
function mpo_to_dense(cores::Vector{Array{Float64,4}})
    N = length(cores)
    _, d1, _, Rr1 = size(cores[1])
    acc = dropdims(cores[1]; dims=1)               # (d1, d1, Rr1)
    Dout, Din, Rb = size(acc)
    @inbounds for i in 2:N
        Rl, di, _, Rr = size(cores[i])
        @assert Rl == Rb "MPO bond mismatch between core $(i-1) and core $i"
        newacc = zeros(Float64, Dout * di, Din * di, Rr)
        for a in 1:Rb, i1 in 1:Din, o1 in 1:Dout
            va = acc[o1, i1, a]
            iszero(va) && continue
            for b in 1:Rr, i2 in 1:di, o2 in 1:di
                w = cores[i][a, o2, i2, b]
                iszero(w) && continue
                newacc[(o1 - 1) * di + o2, (i1 - 1) * di + i2, b] += va * w
            end
        end
        acc = newacc
        Dout, Din, Rb = size(acc)
    end
    @assert Rb == 1 "MPO did not close to a scalar right boundary"
    return dropdims(acc; dims=3)
end

# ── Sparse ground/excited states (Lanczos via KrylovKit) ──────────────────────

"""
    exact_ground_states(H; nev=2, rng=Random.default_rng()) -> (vals, vecs)

Smallest `nev` eigenpairs of the symmetric sparse Hamiltonian `H` via Lanczos.
`vecs[n]` is normalised; the ground state (Perron-Frobenius, Thm. 4.4) is
sign-fixed so that its largest-magnitude component is positive.
"""
function exact_ground_states(H::AbstractMatrix; nev::Int=2, rng=Random.default_rng())
    D = size(H, 1)
    x0 = randn(rng, D)
    vals, vecs, info = eigsolve(H, x0, nev, :SR; ishermitian=true, tol=1e-12, krylovdim=max(2nev + 10, 30), maxiter=500)
    @assert info.converged >= nev "Lanczos: only $(info.converged)/$nev eigenpairs converged"
    out_vals = Float64.(vals[1:nev])
    out_vecs = Vector{Vector{Float64}}(undef, nev)
    for n in 1:nev
        v = Float64.(vecs[n])
        v ./= norm(v)
        j = argmax(abs.(v))
        v[j] < 0 && (v .*= -1)
        out_vecs[n] = v
    end
    return out_vals, out_vecs
end

# ── TT-SVD (report Thm. 6.2 / Appendix A) ──────────────────────────────────────

"""
    tt_svd(tensor; maxdim=nothing) -> (cores, singvals)

Sequential left-to-right TT-SVD of a dense order-`N` tensor into an MPS in
`(Dl,d,Dr)` convention (matching `MPSFast.core.jl`). If `maxdim` is given,
every bond is truncated to at most `maxdim` singular values (report §6.1,
Thm. 6.2 / Prop. 6.3).
"""
function tt_svd(tensor::AbstractArray{Float64}; maxdim::Union{Nothing,Int}=nothing)
    dims = collect(size(tensor))
    N = length(dims)
    cores = Vector{Array{Float64,3}}(undef, N)
    svals = Vector{Vector{Float64}}(undef, N - 1)
    r_prev = 1
    C = reshape(tensor, r_prev * dims[1], div(length(tensor), r_prev * dims[1]))
    for kk in 1:(N - 1)
        F = svd(C)
        r = length(F.S)
        maxdim !== nothing && (r = min(r, maxdim))
        U = F.U[:, 1:r]
        S = F.S[1:r]
        Vt = F.Vt[1:r, :]
        cores[kk] = reshape(U, r_prev, dims[kk], r)
        svals[kk] = S
        C2 = Diagonal(S) * Vt                      # (r, rest)
        if kk < N - 1
            rest = div(size(C2, 2), dims[kk + 1])
            C = reshape(C2, r * dims[kk + 1], rest)
        else
            cores[N] = reshape(C2, r, dims[N], 1)
        end
        r_prev = r
    end
    return cores, svals
end

"""
Reconstruct the dense order-`N` tensor `ψ[σ1,...,σN]` (Julia's native
column-major axis order: site 1 = axis 1, varies fastest in `vec(...)`)
from MPS cores `(Dl,d,Dr)`. Use [`mps_to_lex_vector`](@ref) to convert to
the report's lexicographic (site-1-slowest) flat-vector convention used by
`build_hamiltonian_sparse` / `config_to_index`.
"""
function mps_to_tensor(cores::Vector{Array{Float64,3}})
    N = length(cores)
    dims = [size(A, 2) for A in cores]
    d0, Dr0 = size(cores[1], 2), size(cores[1], 3)
    acc = reshape(cores[1], d0, Dr0)                # (prod_so_far, Dl_current)
    @inbounds for j in 2:N
        Dl, d, Dr = size(cores[j])
        Amat = reshape(cores[j], Dl, d * Dr)
        newmat = acc * Amat                         # (prod_so_far, d*Dr)
        acc = reshape(newmat, size(newmat, 1) * d, Dr)
    end
    @assert size(acc, 2) == 1
    return reshape(acc, dims...)
end

"""Flatten an MPS to a plain vector in Julia's native (site-1-fastest) order."""
mps_to_vector(cores::Vector{Array{Float64,3}}) = vec(mps_to_tensor(cores))

# ── Lexicographic ⇔ natural tensor conversion (explicit, unambiguous) ─────────
#
# `build_hamiltonian_sparse` / `config_to_index` use a lexicographic flat
# index with site 1 slowest-varying (report Appendix A / eq. 22 ordering).
# `tt_svd` / `mps_to_tensor` operate on plain Julia arrays, whose native
# `vec(...)` order has axis 1 (site 1) *fastest*-varying. These two helpers
# convert between the conventions via explicit index loops (unambiguous,
# and cheap relative to the eigensolver for the sizes used here).

function _lex_strides(dims::Vector{Int})
    N = length(dims)
    strides = ones(Int, N)
    @inbounds for i in (N - 1):-1:1
        strides[i] = strides[i + 1] * dims[i + 1]
    end
    return strides
end

"""Lexicographic flat vector (site 1 slowest) → natural tensor `T[σ1,...,σN]`."""
function lex_vector_to_tensor(v::AbstractVector{T}, dims::Vector{Int}) where {T}
    N = length(dims)
    strides = _lex_strides(dims)
    out = Array{T}(undef, dims...)
    @inbounds for lin0 in 0:(length(v) - 1)
        rem = lin0
        idx = Vector{Int}(undef, N)
        for i in 1:N
            idx[i] = div(rem, strides[i]) + 1
            rem = rem % strides[i]
        end
        out[idx...] = v[lin0 + 1]
    end
    return out
end

"""Natural tensor `T[σ1,...,σN]` → lexicographic flat vector (site 1 slowest)."""
function tensor_to_lex_vector(Tarr::AbstractArray{T}, dims::Vector{Int}) where {T}
    strides = _lex_strides(dims)
    v = Vector{T}(undef, prod(dims))
    @inbounds for idx in CartesianIndices(Tarr)
        lin0 = 0
        for i in 1:length(dims)
            lin0 += (idx[i] - 1) * strides[i]
        end
        v[lin0 + 1] = Tarr[idx]
    end
    return v
end

"""Reconstruct MPS cores directly into the report's lexicographic flat-vector convention."""
function mps_to_lex_vector(cores::Vector{Array{Float64,3}}, dims::Vector{Int})
    return tensor_to_lex_vector(mps_to_tensor(cores), dims)
end

"""Von Neumann entropy (nats) of each bipartite cut from exact (untruncated) TT-SVD."""
function bipartite_entropies_exact(tensor::AbstractArray{Float64})
    _, svals = tt_svd(tensor; maxdim=nothing)
    entropies = Vector{Float64}(undef, length(svals))
    for (b, s) in enumerate(svals)
        p = abs2.(s)
        z = sum(p)
        entropies[b] = z > 0 ? sum(-(pk / z) * log(pk / z) for pk in p if pk / z > 1e-30) : NaN
    end
    return entropies
end

# ── Quote-ratio metrics (report §6.4, §8.3) ────────────────────────────────────

"""
    central_grid_edges(N) -> Vector{NamedTuple}
    central_grid_edges(Qs::Vector{Int}) -> Vector{NamedTuple}

Directed edges `(q, i, dir)` on the central region `q_i ∈ {-1,0,1}`, keeping only
jumps that stay inside `[-Q_i, Q_i]` (report §8.3 when `Q_i ≥ 1`).
"""
function central_grid_edges(Qs::Vector{Int})
    N = length(Qs)
    edges = NamedTuple{(:q, :i, :dir),Tuple{Vector{Int},Int,Int}}[]
    for combo in Iterators.product(fill((-1, 0, 1), N)...)
        q = collect(combo)
        for i in 1:N, dir in (1, -1)
            qn = copy(q)
            qn[i] += dir
            in_bounds(q, Qs) && in_bounds(qn, Qs) && push!(edges, (; q, i, dir))
        end
    end
    return edges
end

central_grid_edges(N::Int) = central_grid_edges(fill(2, N))

@inline function in_bounds(q::Vector{Int}, Qs::Vector{Int})
    return all(-Qs[i] <= q[i] <= Qs[i] for i in eachindex(Qs))
end

"""
    quote_log_ratio_errors(ϕ_approx, ϕ_exact, Qs, edges) -> (rmse, maxabs)

RMSE and max-abs error of `log ϕ(q) − log ϕ(q ± e_i)` between an approximate
and the exact (both strictly positive, same sign convention) amplitude
vector over the given directed edges (report §6.4, Lemma 6.6; §8.3).
"""
function quote_log_ratio_errors(
    ϕ_approx::AbstractVector{<:Real}, ϕ_exact::AbstractVector{<:Real},
    Qs::Vector{Int}, edges,
)
    se = 0.0
    maxabs = 0.0
    n = 0
    for e in edges
        qn = copy(e.q)
        qn[e.i] += e.dir
        i0 = config_to_index(e.q, Qs)
        i1 = config_to_index(qn, Qs)
        r_exact = log(ϕ_exact[i0]) - log(ϕ_exact[i1])
        r_approx = log(abs(ϕ_approx[i0])) - log(abs(ϕ_approx[i1]))
        err = r_approx - r_exact
        se += err^2
        maxabs = max(maxabs, abs(err))
        n += 1
    end
    return sqrt(se / n), maxabs
end

"""Sign-align (to `ϕ_ref`) and normalise an approximate state vector."""
function align_and_normalize(ϕ::AbstractVector{<:Real}, ϕ_ref::AbstractVector{<:Real})
    v = Float64.(ϕ) ./ norm(ϕ)
    dot(v, ϕ_ref) < 0 && (v .*= -1)
    return v
end

export MarketMakingModel, c_coeff, site_dims, hilbert_dim
export config_to_index, index_to_config, local_shift_ops, kron_at
export build_hamiltonian_sparse, build_mpo_cores, mpo_to_dense
export exact_ground_states, tt_svd, mps_to_vector, mps_to_tensor, mps_to_lex_vector
export lex_vector_to_tensor, tensor_to_lex_vector, bipartite_entropies_exact
export central_grid_edges, quote_log_ratio_errors, align_and_normalize
