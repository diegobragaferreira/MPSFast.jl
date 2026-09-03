# qubo.jl — QUBO / Ising combinatorial problems and reference solvers.
#
# Binary variables x ∈ {0,1}^n minimise
#   E(x) = offset + h'x + x'Q x
# with symmetric Q (upper-triangle convention in `Q`; diagonal stored in Q[i,i]
# and also available via `h` after `QUBOProblem` construction).

using Random

"""Binary quadratic unconstrained optimisation problem."""
struct QUBOProblem{T<:Real}
    Q::Matrix{T}   # n×n symmetric
    h::Vector{T}   # linear coefficients
    offset::T
end

function QUBOProblem(Q::AbstractMatrix{T}, h::AbstractVector{T}=zeros(T, size(Q, 1)); offset::T=zero(T)) where {T<:Real}
    n = size(Q, 1)
    @assert size(Q, 2) == n && length(h) == n
    return QUBOProblem(Matrix{T}(0.5 * (Q + Q')), Vector{T}(h), offset)
end

"""Number of binary variables."""
nvars(prob::QUBOProblem) = length(prob.h)

"""
    qubo_energy(prob, x) -> Float64

Evaluate `E(x)` for `x` a `BitVector`, `Vector{Bool}`, or `Vector{Int}` in `{0,1}`.
"""
function qubo_energy(prob::QUBOProblem, x::AbstractVector{<:Integer})
    n = nvars(prob)
    @assert length(x) == n
    E = float(prob.offset)
    @inbounds for i in 1:n
        xi = x[i]
        (xi == 0 || xi == 1) || throw(ArgumentError("qubo_energy: entries must be 0 or 1"))
        E += prob.h[i] * xi
        E += prob.Q[i, i] * xi
        for j in (i + 1):n
            xj = x[j]
            E += prob.Q[i, j] * xi * xj
        end
    end
    return E
end

qubo_energy(prob::QUBOProblem, x::BitVector) = qubo_energy(prob, collect(Int, x))

"""1D chain QUBO: `E(x) = Σ h[i] x_i + Σ J[i] x_i x_{i+1}` (binary `x`)."""
struct ChainQUBOProblem{T<:Real}
    h::Vector{T}
    J::Vector{T}   # length n-1
    offset::T
end

ChainQUBOProblem(h::AbstractVector{T}, J::AbstractVector{T}; offset::T=zero(T)) where {T<:Real} =
    ChainQUBOProblem(Vector{T}(h), Vector{T}(J), offset)

nvars(prob::ChainQUBOProblem) = length(prob.h)

function qubo_energy(prob::ChainQUBOProblem, x::AbstractVector{<:Integer})
    n = nvars(prob)
    @assert length(x) == n
    E = float(prob.offset)
    @inbounds for i in 1:n
        E += prob.h[i] * x[i]
    end
    @inbounds for i in 1:(n - 1)
        E += prob.J[i] * x[i] * x[i + 1]
    end
    return E
end

qubo_energy(prob::ChainQUBOProblem, x::BitVector) = qubo_energy(prob, collect(Int, x))

"""Convert a chain problem to dense `QUBOProblem` (identity variable order)."""
function dense_qubo(prob::ChainQUBOProblem{T}) where {T<:Real}
    n = nvars(prob)
    Q = zeros(T, n, n)
    h = copy(prob.h)
    @inbounds for i in 1:(n - 1)
        Q[i, i + 1] = prob.J[i]
        Q[i + 1, i] = prob.J[i]
    end
    return QUBOProblem(Q, h; offset=prob.offset)
end

"""
    chain_qubo_from_dense(Q; h=zeros(n), order=nothing) -> ChainQUBOProblem
    chain_qubo_from_dense(prob::QUBOProblem; order=nothing) -> ChainQUBOProblem

Extract nearest-neighbour couplings along `order` (default `1:n`).
Off-diagonal `Q[i,j]` with `|i-j|>1` is dropped.
"""
function chain_qubo_from_dense(
    Q::AbstractMatrix{T},
    h::AbstractVector{T}=zeros(T, size(Q, 1));
    order::Union{Nothing,Vector{Int}}=nothing,
    offset::T=zero(T),
) where {T<:Real}
    n = size(Q, 1)
    order === nothing && (order = collect(1:n))
    @assert length(order) == n && length(h) == n
    hc = [T(h[order[i]]) for i in 1:n]
    J = T[ Q[order[i], order[i + 1]] for i in 1:(n - 1) ]
    return ChainQUBOProblem(hc, J; offset=offset)
end

chain_qubo_from_dense(prob::QUBOProblem; order::Union{Nothing,Vector{Int}}=nothing) =
    chain_qubo_from_dense(prob.Q, prob.h; order=order, offset=prob.offset)

# ─── Problem builders ─────────────────────────────────────────────────────────

"""
    random_qubo(n; rng, scale) -> QUBOProblem

Random symmetric `Q` and linear `h` (spin-glass style benchmark).
"""
function random_qubo(n::Int; rng::AbstractRNG=Random.default_rng(), scale::Real=1.0)
    R = scale * randn(rng, n, n)
    Q = Matrix{Float64}(0.5 * (R + R'))
    h = scale * randn(rng, n)
    return QUBOProblem(Q, h)
end

"""
    maxcut_qubo(adj::AbstractMatrix) -> QUBOProblem

MaxCut on weighted graph: maximise `Σ_{i<j} w_ij (x_i + x_j - 2 x_i x_j)`.
"""
function maxcut_qubo(adj::AbstractMatrix{<:Real})
    n = size(adj, 1)
    @assert size(adj, 2) == n
    Q = zeros(Float64, n, n)
    h = zeros(Float64, n)
    offset = 0.0
    @inbounds for i in 1:n
        for j in (i + 1):n
            w = adj[i, j]
            iszero(w) && continue
            # cut value term: w*(x_i + x_j - 2 x_i x_j) = w*x_i + w*x_j - 2w*x_i*x_j
            h[i] += w
            h[j] += w
            Q[i, j] -= 2w
            offset += w   # constant w per edge when x=0,0
        end
    end
    return QUBOProblem(Q, h; offset=offset)
end

"""Path-graph MaxCut (chain topology) — exact chain DP applies."""
maxcut_path_qubo(n::Int; w::Real=1.0) = maxcut_qubo(
    diagm(-1 => fill(w, n - 1), 1 => fill(w, n - 1)))

"""
    number_partition_qubo(s::AbstractVector) -> QUBOProblem

Partition integers `s` into two groups minimising `(Σ_+ - Σ_-)^2`.
"""
function number_partition_qubo(s::AbstractVector{<:Real})
    n = length(s)
    Q = zeros(Float64, n, n)
    h = copy(Float64.(s))
    @inbounds for i in 1:n
        Q[i, i] = -Float64(s[i])^2
        for j in (i + 1):n
            Q[i, j] = 2 * Float64(s[i]) * Float64(s[j])
        end
    end
    return QUBOProblem(Q, h)
end

"""
    portfolio_selection_qubo(μ, Σ; λ, target_k, penalty) -> QUBOProblem

Binary asset selection: minimise `-μ'x + λ x'Σx + penalty*(1'x - target_k)^2`.
"""
function portfolio_selection_qubo(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    λ::Real=1.0,
    target_k::Int=length(μ) ÷ 2,
    penalty::Real=10.0,
)
    n = length(μ)
    @assert size(Σ) == (n, n)
    Q = λ * Matrix{Float64}(Σ)
    h = -Float64.(μ)
    # (sum x - k)^2 = sum x_i + 2*sum_{i<j} x_i x_j - 2k*sum x_i + k^2  (binary x)
    @inbounds for i in 1:n
        Q[i, i] += penalty
        h[i] -= 2 * penalty * target_k
        for j in (i + 1):n
            Q[i, j] += 2 * penalty
            Q[j, i] += 2 * penalty
        end
    end
    offset = Float64(penalty * target_k^2)
    return QUBOProblem(Q, h; offset=offset)
end

"""Markowitz subset energy `-μ'x + λ x'Σx` for binary `x`."""
function portfolio_markowitz_energy(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    x::AbstractVector{<:Integer};
    λ::Real=1.0,
)
    xf = Float64.(x)
    return -dot(μ, xf) + λ * dot(xf, Σ, xf)
end

function _foreach_k_subset(n::Int, k::Int, f::Function)
    k == 0 && return
    c = collect(1:k)
    while true
        f(c)
        i = k
        while i >= 1 && c[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        c[i] += 1
        @inbounds for j in (i + 1):k
            c[j] = c[j - 1] + 1
        end
    end
end

"""
    optimize_portfolio_exact_k(μ, Σ; k, λ) -> NamedTuple

Exact cardinality-constrained Markowitz: minimise `-μ'x + λ x'Σx` over all `x` with `sum(x)=k`.
Feasible for moderate `n` (enumerates `C(n,k)` subsets).
"""
function optimize_portfolio_exact_k(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    k::Int,
    λ::Real=1.0,
)
    n = length(μ)
    @assert size(Σ) == (n, n)
    @assert 0 <= k <= n
    k == 0 && return (; x=zeros(Int, n), energy=0.0)
    best_E = Inf
    best_x = zeros(Int, n)
    _foreach_k_subset(n, k, function(idx)
        x = zeros(Int, n)
        @inbounds for j in idx
            x[j] = 1
        end
        E = portfolio_markowitz_energy(μ, Σ, x; λ=λ)
        if E < best_E
            best_E = E
            best_x = x
        end
    end)
    return (; x=best_x, energy=best_E)
end

# ─── Reference solvers ────────────────────────────────────────────────────────

"""Exhaustive search (feasible for `n ≤ 20`)."""
function qubo_brute_force(prob::Union{QUBOProblem, ChainQUBOProblem})
    n = nvars(prob)
    n > 20 && @warn "qubo_brute_force: n=$n is large; consider simulated annealing"
    best_x = zeros(Int, n)
    best_E = Inf
    for bits in 0:(2^n - 1)
        x = [(bits >> (i - 1)) & 1 for i in 1:n]
        E = qubo_energy(prob, x)
        if E < best_E
            best_E = E
            best_x = x
        end
    end
    return (; x=best_x, energy=best_E)
end

"""Single-bit flip hill climbing."""
function qubo_local_search(
    prob::Union{QUBOProblem, ChainQUBOProblem};
    x0::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    max_passes::Int=10_000,
)
    n = nvars(prob)
    x = x0 === nothing ? rand(rng, 0:1, n) : collect(Int, x0)
    E = qubo_energy(prob, x)
    for _ in 1:max_passes
        improved = false
        for i in 1:n
            x[i] = 1 - x[i]
            E2 = qubo_energy(prob, x)
            if E2 < E
                E = E2
                improved = true
            else
                x[i] = 1 - x[i]
            end
        end
        improved || break
    end
    return (; x, energy=E)
end

"""Metropolis simulated annealing."""
function qubo_simulated_annealing(
    prob::Union{QUBOProblem, ChainQUBOProblem};
    rng::AbstractRNG=Random.default_rng(),
    n_steps::Int=50_000,
    T0::Real=10.0,
    T1::Real=0.01,
)
    n = nvars(prob)
    x = rand(rng, 0:1, n)
    E = qubo_energy(prob, x)
    best_x = copy(x)
    best_E = E
    for t in 1:n_steps
        T = T0 * (T1 / T0)^(t / n_steps)
        i = rand(rng, 1:n)
        x[i] = 1 - x[i]
        E2 = qubo_energy(prob, x)
        Δ = E2 - E
        if Δ <= 0 || rand(rng) < exp(-Δ / T)
            E = E2
            if E < best_E
                best_E = E
                best_x = copy(x)
            end
        else
            x[i] = 1 - x[i]
        end
    end
    return (; x=best_x, energy=best_E)
end

"""
    chain_qubo_exact_dp(prob::ChainQUBOProblem) -> (x, energy)

Exact O(n) dynamic programme for 1D chain QUBO.
"""
function chain_qubo_exact_dp(prob::ChainQUBOProblem)
    n = nvars(prob)
    n == 0 && return (; x=Int[], energy=float(prob.offset))
    V = Matrix{Float64}(undef, n, 2)   # V[i, xi+1] for xi ∈ {0,1}
    back = Matrix{Int}(undef, n, 2)
    V[1, 1] = 0.0
    V[1, 2] = prob.h[1]
    @inbounds for i in 2:n
        for xi in 0:1
            best = Inf
            arg = 0
            for xp in 0:1
                e = V[i - 1, xp + 1] + prob.h[i] * xi + prob.J[i - 1] * xp * xi
                if e < best
                    best = e
                    arg = xp
                end
            end
            V[i, xi + 1] = best
            back[i, xi + 1] = arg
        end
    end
    e0, e1 = V[n, 1], V[n, 2]
    xn = e0 <= e1 ? 0 : 1
    x = zeros(Int, n)
    x[n] = xn
    @inbounds for i in n:-1:2
        x[i - 1] = back[i, x[i] + 1]
    end
    return (; x, energy=qubo_energy(prob, x))
end

# ─── Ising conversions ────────────────────────────────────────────────────────

"""
    ising_to_qubo(h, J; periodic=false) -> ChainQUBOProblem

Spin variables `s ∈ {-1,+1}` with `H = -Σ h_i s_i - Σ J_i s_i s_{i+1}`.
Mapped to `x = (1+s)/2 ∈ {0,1}`.
"""
function ising_to_qubo(
    h::AbstractVector{<:Real},
    J::AbstractVector{<:Real};
    periodic::Bool=false,
)
    n = length(h)
    if n == 1
        return ChainQUBOProblem([-Float64(h[1])]; offset=-Float64(h[1]))
    end
    periodic || @assert length(J) == n - 1
    periodic && @assert length(J) == n
    h2 = zeros(Float64, n)
    J2 = zeros(Float64, n - 1)
    offset = 0.0
    @inbounds for i in 1:n
        hi = Float64(h[i])
        h2[i] -= hi
        offset -= hi
    end
    pairs = periodic ? [(i, mod1(i + 1, n)) for i in 1:n] : [(i, i + 1) for i in 1:(n - 1)]
    @inbounds for (k, (i, j)) in enumerate(pairs)
        Ji = Float64(J[k])
        J2[k] = -Ji
        h2[i] -= Ji
        h2[j] -= Ji
        offset += Ji
    end
    return ChainQUBOProblem(h2, J2; offset=offset)
end

"""Decode physical MPS index `σ ∈ {1,2}` to binary `x ∈ {0,1}`."""
@inline mps_bit_index(σ::Integer) = σ - 1

"""Encode binary `x` to MPS physical index."""
@inline bit_mps_index(x::Integer) = x + 1
