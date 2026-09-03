# portfolio_loop.jl — Generative scenarios → moment estimates → QUBO portfolio selection.

using Random
using LinearAlgebra
using Printf

"""
    estimate_return_moments(R) -> (μ, Σ)

`R` is `n_scenarios × n_assets` (each row one scenario, each column one asset).
"""
function estimate_return_moments(R::AbstractMatrix{<:Real})
    n, p = size(R)
    n == 0 && return zeros(p), zeros(p, p)
    μ = [sum(R[:, j]) / n for j in 1:p]
    Σ = zeros(Float64, p, p)
    denom = max(n - 1, 1)
    @inbounds for i in 1:p, j in 1:p
        s = 0.0
        for r in 1:n
            s += (R[r, i] - μ[i]) * (R[r, j] - μ[j])
        end
        Σ[i, j] = s / denom
    end
    return Float64.(μ), Σ
end

"""
    gaussian_return_scenarios(n_scenarios, μ, Σ; rng) -> Matrix

Sample `n_scenarios × n_assets` returns from 𝒩(μ, Σ) (MC baseline generative model).
"""
function gaussian_return_scenarios(
    n_scenarios::Int,
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    rng::AbstractRNG=Random.default_rng(),
)
    p = length(μ)
    @assert size(Σ) == (p, p)
    L = cholesky(Symmetric(Matrix{Float64}(Σ) + 1e-10I))
    Z = randn(rng, p, n_scenarios)
    X = μ .+ L.L * Z
    return Matrix{Float64}(X')   # n_scenarios × p
end

"""
    factor_return_scenarios(factor_returns, β, σ_idio; rng) -> Matrix

Build multivariate scenarios from a market-factor sample:
`r_{s,i} = β_i f_s + σ_idio_i ε_{s,i}`.
"""
function factor_return_scenarios(
    factor_returns::AbstractVector{<:Real},
    β::AbstractVector{<:Real},
    σ_idio::AbstractVector{<:Real};
    rng::AbstractRNG=Random.default_rng(),
)
    n = length(factor_returns)
    p = length(β)
    @assert length(σ_idio) == p
    R = Matrix{Float64}(undef, n, p)
    @inbounds for s in 1:n
        f = factor_returns[s]
        for i in 1:p
            R[s, i] = β[i] * f + σ_idio[i] * randn(rng)
        end
    end
    return R
end

"""Log return `log(S_T / S_0)` per row of price paths (`n_paths × M`)."""
function path_log_returns(paths::AbstractMatrix{<:Real})
    n = size(paths, 1)
    out = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        s0 = paths[s, 1]
        sT = paths[s, end]
        out[s] = log(max(sT, 1e-12) / max(s0, 1e-12))
    end
    return out
end

"""
    train_factor_mps(
        train_paths;
        m=3, D_max=16, n_epochs=30, η=0.05, ε_cut=1e-10, rng,
    ) -> (mps, enc)

Quick Born-machine fit for a **single** price-factor series (`n_paths × M`).
"""
function train_factor_mps(
    train_paths::AbstractMatrix{<:Real};
    m::Int=3,
    D_max::Int=16,
    n_epochs::Int=30,
    η::Real=0.05,
    ε_cut::Real=1e-10,
    rng::AbstractRNG=Random.default_rng(),
)
    enc = Encoders.BasisEncoder(m)
    Encoders.fit_grid!(enc, train_paths)
    xi = Encoders.encode_paths(enc, train_paths)
    M = Encoders.chain_length(enc, size(train_paths, 2))
    d = Encoders.site_dim(enc)
    mps = init_mps(M, d, D_max; rng=rng)
    train_mps!(mps, xi, n_epochs, η, D_max, ε_cut; verbose=false)
    return mps, enc
end

"""
    mps_factor_return_scenarios(
        mps, enc, β, σ_idio, n_scenarios; seed, rng,
    ) -> Matrix

Sample price paths from a trained factor MPS, map each to a log return,
then expand to multivariate returns via `factor_return_scenarios`.
"""
function mps_factor_return_scenarios(
    mps,
    enc::Encoders.PathEncoder,
    β::AbstractVector{<:Real},
    σ_idio::AbstractVector{<:Real},
    n_scenarios::Int;
    seed::Int=0,
    rng::AbstractRNG=Random.default_rng(),
)
    paths, _ = Encoders.sample_paths(enc, mps, n_scenarios; seed=seed)
    f = path_log_returns(paths)
    return factor_return_scenarios(f, β, σ_idio; rng=rng)
end

"""
    generative_portfolio_qubo_loop(
        R_train;
        λ, target_k, penalty, qubo_method, rng, qubo_kwargs...,
    )

1. Estimate `μ̂, Σ̂` from scenario matrix `R_train` (`n × p`).
2. Build `portfolio_selection_qubo(μ̂, Σ̂)`.
3. Solve with `optimize_qubo`.
"""
function generative_portfolio_qubo_loop(
    R_train::AbstractMatrix{<:Real};
    λ::Real=1.0,
    target_k::Int=size(R_train, 2) ÷ 2,
    penalty::Real=10.0,
    qubo_method::Symbol=:exact_k,
    rng::AbstractRNG=Random.default_rng(),
    qubo_kwargs...,
)
    μ_hat, Σ_hat = estimate_return_moments(R_train)
    prob = portfolio_selection_qubo(μ_hat, Σ_hat; λ=λ, target_k=target_k, penalty=penalty)
    sol = if qubo_method == :exact_k
        optimize_portfolio_exact_k(μ_hat, Σ_hat; k=target_k, λ=λ)
    else
        optimize_qubo(prob; method=qubo_method, rng=rng, qubo_kwargs...)
    end
    return (;
        prob,
        sol,
        μ_hat,
        Σ_hat,
        selected=findall(==(1), sol.x),
        n_selected=sum(sol.x),
    )
end

"""
    equal_weight_portfolio_stats(x, R) -> NamedTuple

Binary mask `x` (0/1); equal weight on selected assets. `R` is `n_scenarios × n_assets`.
"""
function equal_weight_portfolio_stats(
    x::AbstractVector{<:Integer},
    R::AbstractMatrix{<:Real},
)
    sel = findall(==(1), x)
    isempty(sel) && return (; mean=NaN, std=NaN, sharpe=NaN, n=0)
    w = 1.0 / length(sel)
    port = [sum(R[s, i] * w for i in sel) for s in 1:size(R, 1)]
    μ = sum(port) / length(port)
    σ = sqrt(sum((r - μ)^2 for r in port) / max(length(port) - 1, 1))
    sharpe = σ > 0 ? μ / σ : NaN
    return (; mean=μ, std=σ, sharpe=sharpe, n=length(sel))
end

"""
    compare_generative_portfolio_sources(
        R_train_mps, R_train_ref, R_oos;
        λ, target_k, penalty, qubo_method, rng,
    )

Run the QUBO portfolio loop on two training scenario sets (e.g. MPS vs Gaussian),
then evaluate equal-weight OOS stats on `R_oos`.
"""
function compare_generative_portfolio_sources(
    R_train_mps::AbstractMatrix{<:Real},
    R_train_ref::AbstractMatrix{<:Real},
    R_oos::AbstractMatrix{<:Real};
    λ::Real=1.0,
    target_k::Int=size(R_train_mps, 2) ÷ 2,
    penalty::Real=10.0,
    qubo_method::Symbol=:exact_k,
    rng::AbstractRNG=Random.default_rng(),
)
    mps_res = generative_portfolio_qubo_loop(
        R_train_mps; λ=λ, target_k=target_k, penalty=penalty,
        qubo_method=qubo_method, rng=rng)
    ref_res = generative_portfolio_qubo_loop(
        R_train_ref; λ=λ, target_k=target_k, penalty=penalty,
        qubo_method=qubo_method, rng=rng)

    mps_oos = equal_weight_portfolio_stats(mps_res.sol.x, R_oos)
    ref_oos = equal_weight_portfolio_stats(ref_res.sol.x, R_oos)

    return (;
        mps=(; train=mps_res, oos=mps_oos),
        ref=(; train=ref_res, oos=ref_oos),
    )
end

"""
    simulate_random_walk_prices(N, M; S0, σ, drift, rng) -> Matrix

Toy geometric random walk for factor MPS training (`N × M`).
"""
function simulate_random_walk_prices(
    N::Int, M::Int;
    S0::Real=100.0,
    σ::Real=0.02,
    drift::Real=0.0,
    rng::AbstractRNG=Random.default_rng(),
)
    paths = Matrix{Float64}(undef, N, M)
    @inbounds for i in 1:N
        s = Float64(S0)
        for t in 1:M
            s *= exp(drift + σ * randn(rng))
            paths[i, t] = s
        end
    end
    return paths
end

"""
    demo_generative_portfolio_loop(;
        n_assets, n_train, n_oos, target_k, M, mps_epochs, rng,
    ) -> NamedTuple

End-to-end smoke/demo: train factor MPS, build scenarios, QUBO select, OOS eval vs Gaussian.
"""
function demo_generative_portfolio_loop(;
    n_assets::Int=6,
    n_train::Int=2_000,
    n_oos::Int=500,
    target_k::Int=3,
    M::Int=10,
    m::Int=3,
    D_max::Int=16,
    mps_epochs::Int=25,
    λ::Real=0.5,
    penalty::Real=40.0,
    rng::AbstractRNG=Random.default_rng(),
)
    p = n_assets
    β = 0.6 .+ 0.3 .* rand(rng, p)
    σ_idio = 0.01 .+ 0.02 .* rand(rng, p)
    μ_true = 0.001 .* randn(rng, p)
    A = randn(rng, p, p)
    Σ_true = A' * A / p + 0.001 * I

    train_paths = simulate_random_walk_prices(n_train, M; rng=rng)
    mps, enc = train_factor_mps(
        train_paths; m=m, D_max=D_max, n_epochs=mps_epochs, rng=rng)

    R_train_mps = mps_factor_return_scenarios(
        mps, enc, β, σ_idio, n_train; seed=1, rng=rng)
    R_train_ref = gaussian_return_scenarios(n_train, μ_true, Σ_true; rng=rng)
    R_oos = gaussian_return_scenarios(n_oos, μ_true, Σ_true; rng=rng)

    cmp = compare_generative_portfolio_sources(
        R_train_mps, R_train_ref, R_oos;
        λ=λ, target_k=target_k, penalty=penalty,
        qubo_method=:exact_k, rng=rng)

    return (;
        mps, enc, β, σ_idio, μ_true, Σ_true,
        R_train_mps, R_train_ref, R_oos, cmp,
    )
end
