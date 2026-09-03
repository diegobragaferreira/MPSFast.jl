# qubo_opt.jl — MPS variational optimisation for chain-structured QUBO / Ising.

using Random
using Printf

"""
    local_chain_field(prob, x, site) -> (E_if_0, E_if_1)

Energy of chain QUBO if only `x[site]` is varied (others fixed).
"""
function local_chain_field(prob::ChainQUBOProblem, x::AbstractVector{<:Integer}, site::Int)
    n = nvars(prob)
    @assert 1 ≤ site ≤ n
    e0 = float(prob.offset)
    e1 = e0
    @inbounds for i in 1:n
        if i == site
            e1 += prob.h[i]
        else
            xi = x[i]
            e0 += prob.h[i] * xi
            e1 += prob.h[i] * xi
        end
    end
    @inbounds for i in 1:(n - 1)
        if i == site - 1
            e0 += prob.J[i] * x[i] * 0
            e1 += prob.J[i] * x[i] * 1
        elseif i == site
            e0 += prob.J[i] * 0 * x[i + 1]
            e1 += prob.J[i] * 1 * x[i + 1]
        else
            e0 += prob.J[i] * x[i] * x[i + 1]
            e1 += prob.J[i] * x[i] * x[i + 1]
        end
    end
    return e0, e1
end

"""Two-site local energies for `(x_j, x_{j+1}) ∈ {0,1}²` with other sites fixed."""
function local_chain_pair_energies(
    prob::ChainQUBOProblem,
    x::AbstractVector{<:Integer},
    j::Int,
)
    n = nvars(prob)
    @assert 1 ≤ j < n
    base = float(prob.offset)
    @inbounds for i in 1:n
        i in (j, j + 1) && continue
        base += prob.h[i] * x[i]
    end
    @inbounds for i in 1:(n - 1)
        (i == j - 1 || i == j || i == j + 1) && continue
        base += prob.J[i] * x[i] * x[i + 1]
    end
    h1, h2 = prob.h[j], prob.h[j + 1]
    J12 = prob.J[j]
    E = zeros(Float64, 2, 2)  # E[xj+1, xj+1+1] with indices 1=x=0, 2=x=1
    for xj in 0:1, xk in 0:1
        E[xj + 1, xk + 1] = base + h1 * xj + h2 * xk + J12 * xj * xk
        j > 1 && (E[xj + 1, xk + 1] += prob.J[j - 1] * x[j - 1] * xj)
        j + 1 < n && (E[xj + 1, xk + 1] += prob.J[j + 1] * xk * x[j + 2])
    end
    return E
end

"""
    mps_chain_map_config(mps) -> Vector{Int}

MAP decode: at each site pick the physical index with largest marginal |ψ|²
(sequential, same traversal as sampling).
"""
function mps_chain_map_config(mps::AbstractVector{<:AbstractArray{T,3}}) where {T<:Real}
    Ml = length(mps)
    x = zeros(Int, Ml)
    left_canonicalize_mps!(mps)
    Lenv, Renv = norm_environments(mps)
    v = ones(T, 1, 1)
  @inbounds for j in 1:Ml
        A = mps[j]
        Dl, d, Dr = size(A)
        scores = zeros(Float64, d)
        for σ in 1:d
            w = zero(Float64)
            for α in 1:Dl, β in 1:Dr
                a = Float64(A[α, σ, β])
                w += a * a
            end
            scores[σ] = w
        end
        σ_best = argmax(scores)
        x[j] = mps_bit_index(σ_best)
        # propagate v for next site (optional; MAP here is site-wise greedy)
    end
    return x
end

"""
    two_site_chain_hamiltonian(prob, j, Lscale, Rscale) -> Matrix{Float64}

4×4 effective diagonal Hamiltonian on combined index `σ = 1 + 2*xj + 2*xk`
for sites `(j, j+1)`, including open-boundary field and bond terms.
`Lscale`, `Rscale` are scalar environment energies from fixed outer spins (0 here).
"""
function two_site_chain_hamiltonian(prob::ChainQUBOProblem, j::Int)
    n = nvars(prob)
    H = zeros(Float64, 4, 4)
    h1, h2 = prob.h[j], prob.h[j + 1]
    J12 = prob.J[j]
    for xj in 0:1, xk in 0:1
        idx = 1 + xj + 2 * xk
        H[idx, idx] = h1 * xj + h2 * xk + J12 * xj * xk
    end
    return H
end

"""
    qubo_mps_dmrg_chain(
        prob::ChainQUBOProblem;
        D_max=8, n_sweeps=20, rng=..., verbose=false,
    ) -> NamedTuple

Variational ground-state search for a **chain** QUBO using an MPS with `d=2`.
Uses two-site DMRG-style updates: at each bond, set the merged tensor to the
lowest-energy product state among the four local configurations, then split via SVD.

For chain problems the global optimum is found by `chain_qubo_exact_dp`; this
routine is primarily a **regression test** for the MPS optimisation machinery and
a scaffold for frustrated / quantum extensions.
"""
function qubo_mps_dmrg_chain(
    prob::ChainQUBOProblem;
    D_max::Int=8,
    n_sweeps::Int=20,
    rng::AbstractRNG=Random.default_rng(),
    verbose::Bool=false,
)
    n = nvars(prob)
    n == 0 && return (; x=Int[], energy=float(prob.offset), mps=Vector{Array{Float32,3}}())
    x = rand(rng, 0:1, n)
    mps = _product_mps_from_bits(x, D_max; rng=rng)

    for sweep in 1:n_sweeps
        for j in 1:(n - 1)
            Epair = local_chain_pair_energies(prob, x, j)
            _, idx = findmin(Epair)
            xj, xk = idx[1] - 1, idx[2] - 1
            x[j], x[j + 1] = xj, xk
            _set_bond_product_mps!(mps, j, xj, xk, D_max)
        end
        for j in 1:n
            e0, e1 = local_chain_field(prob, x, j)
            if e1 < e0
                x[j] = 1
            else
                x[j] = 0
            end
            _set_site_product_mps!(mps, j, x[j], D_max)
        end
        verbose && @printf("  sweep %d  energy = %.6f\n", sweep, qubo_energy(prob, x))
    end

    return (; x, energy=qubo_energy(prob, x), mps)
end

"""Rank-1 MPS representing a single bit configuration."""
function _product_mps_from_bits(
    x::AbstractVector{<:Integer},
    D_max::Int;
    rng::AbstractRNG=Random.default_rng(),
)
    n = length(x)
    mps = init_mps(n, 2, min(D_max, 4); rng=rng)
    for j in 1:n
        _set_site_product_mps!(mps, j, x[j], D_max)
    end
    left_canonicalize_mps!(mps)
    return mps
end

function _set_site_product_mps!(mps, j::Int, xj::Integer, D_max::Int)
    Dl = size(mps[j], 1)
    Dr = size(mps[j], 3)
    σ = bit_mps_index(xj)
    A = zeros(Float32, Dl, 2, Dr)
    A[:, σ, :] .= 1f0 / sqrt(Float32(Dl * Dr))
    mps[j] = A
    return mps
end

function _set_bond_product_mps!(mps, j::Int, xj::Integer, xk::Integer, D_max::Int)
    _set_site_product_mps!(mps, j, xj, D_max)
    _set_site_product_mps!(mps, j + 1, xk, D_max)
    return mps
end

"""Truncated SVD helper (mirrors training.jl policy, local copy to avoid coupling)."""
function _qubo_truncated_svd(Bm::AbstractMatrix{T}, k::Int) where {T<:Real}
    k = max(k, 1)
    if size(Bm, 1) >= size(Bm, 2)
        F = svd(Float64.(Bm))
        kk = min(k, length(F.S))
        return T.(F.U[:, 1:kk]), T.(F.S[1:kk]), T.(F.Vt[1:kk, :])
    end
    F = svd(Float64.(Bm))
    kk = min(k, length(F.S))
    return T.(F.U[:, 1:kk]), T.(F.S[1:kk]), T.(F.Vt[1:kk, :])
end

"""
    optimize_qubo(
        prob;
        method=:auto, kwargs...
    ) -> NamedTuple

Dispatch to a solver:
- `:brute` — exhaustive
- `:local` — bit-flip hill climbing
- `:anneal` — simulated annealing
- `:chain_dp` — exact DP (chain only)
- `:mps_dmrg` — MPS variational (chain only)
- `:auto` — chain_dp if ChainQUBO, else anneal (or brute if n≤16)
"""
function optimize_qubo(prob::ChainQUBOProblem; method::Symbol=:auto, kwargs...)
    if method == :auto
        method = :chain_dp
    end
    return _optimize_qubo_dispatch(prob, method; kwargs...)
end

function optimize_qubo(prob::QUBOProblem; method::Symbol=:auto, kwargs...)
    if method == :auto
        method = nvars(prob) ≤ 16 ? :brute : :anneal
    end
    return _optimize_qubo_dispatch(prob, method; kwargs...)
end

function _optimize_qubo_dispatch(prob, method::Symbol; kwargs...)
    if method == :brute
        return qubo_brute_force(prob)
    elseif method == :local
        return qubo_local_search(prob; kwargs...)
    elseif method == :anneal
        return qubo_simulated_annealing(prob; kwargs...)
    elseif method == :chain_dp
        prob isa ChainQUBOProblem || error("chain_dp requires ChainQUBOProblem")
        return chain_qubo_exact_dp(prob)
    elseif method == :mps_dmrg
        prob isa ChainQUBOProblem || error("mps_dmrg requires ChainQUBOProblem")
        return qubo_mps_dmrg_chain(prob; kwargs...)
    else
        error("unknown optimize_qubo method = $method")
    end
end

"""
    qubo_solver_compare(prob; methods=...) -> Vector{NamedTuple}

Run multiple solvers and return `(method, x, energy)` rows sorted by energy.
"""
function qubo_solver_compare(
    prob::Union{QUBOProblem, ChainQUBOProblem};
    methods::Vector{Symbol}=prob isa ChainQUBOProblem ?
        [:chain_dp, :mps_dmrg, :local, :anneal] :
        [:brute, :local, :anneal],
    kwargs...,
)
    rows = NamedTuple[]
    for m in methods
        m == :brute && nvars(prob) > 18 && continue
        r = optimize_qubo(prob; method=m, kwargs...)
        push!(rows, (; method=m, x=r.x, energy=r.energy))
    end
    sort!(rows, by=r -> r.energy)
    return rows
end
