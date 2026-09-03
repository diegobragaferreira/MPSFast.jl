# market_making_experiments.jl — Coutinho report §10.1 experiment suite:
# phase diagram, finite-horizon evolution, Doob-chain verification,
# Gaussian / product baselines, and site-ordering sweeps.

using LinearAlgebra
using SparseArrays
using Random

# ── Reference instances ───────────────────────────────────────────────────────

"""Report §8 benchmark instance (N=8, Q_i=2, K=2)."""
function report_benchmark_model()
    N = 8
    Qs = fill(2, N)
    k, γ = 1.0, 1.1
    μ = zeros(N)
    Δ = [0.70, 0.75, 0.80, 0.85, 0.85, 0.80, 0.75, 0.70]
    B = [0.30 0.28; 0.34 0.24; 0.38 0.20; 0.42 0.16;
         0.46 -0.16; 0.50 -0.20; 0.54 -0.24; 0.58 -0.28]
    Σ = Diagonal(Δ) + B * B'
    η = [1.00, 0.95, 1.05, 0.90, 1.10, 1.00, 0.92, 1.08]
    model = MarketMakingModel(N, Qs, Matrix(Σ), μ, k, γ, η)
    return model, Δ, B, build_mpo_cores(model, Δ, B)
end

# ── Random synthetic instances ────────────────────────────────────────────────

"""
    random_factor_model(N, K; Q, factor_strength, rng) -> (model, Δ, B, cores)

Build a random diagonal-plus-factor instance Σ = Δ + λ B Bᵀ with heterogeneous
`η`, zero drift, and `factor_strength` λ controlling off-diagonal coupling.
"""
function random_factor_model(
    N::Int, K::Int;
    Q::Int=2,
    factor_strength::Float64=1.0,
    k::Float64=1.0,
    γ::Float64=1.1,
    rng::AbstractRNG=Random.default_rng(),
)
    Qs = fill(Q, N)
    Δ = 0.6 .+ 0.3 .* rand(rng, N)
    Braw = randn(rng, N, K)
    B = factor_strength .* Braw ./ max(norm(Braw), 1e-12)
    Σ = Diagonal(Δ) + B * B'
    η = 0.9 .+ 0.2 .* rand(rng, N)
    μ = zeros(N)
    model = MarketMakingModel(N, Qs, Matrix(Σ), μ, k, γ, η)
    cores = build_mpo_cores(model, Δ, B)
    return model, Δ, B, cores
end

"""Permute asset order: returns `(model_perm, Δp, Bp, cores, perm)`."""
function permute_factor_model(
    model::MarketMakingModel, Δ::Vector{Float64}, B::Matrix{Float64}, perm::Vector{Int},
)
    N = model.N
    @assert length(perm) == N && sort(perm) == collect(1:N)
    Σp = model.Σ[perm, perm]
    Δp = Δ[perm]
    Bp = B[perm, :]
    ηp = model.η[perm]
    Qsp = model.Qs[perm]
    mp = MarketMakingModel(N, Qsp, Matrix(Σp), model.μ[perm], model.k, model.γ, ηp)
    return mp, Δp, Bp, build_mpo_cores(mp, Δp, Bp), perm
end

"""Greedy seriation: order assets by descending diagonal of Σ (proxy for clustering)."""
function seriation_order(Σ::Matrix{Float64})
    return sortperm(diag(Σ); rev=true)
end

"""Correlation-clustering proxy: greedy nearest-neighbour chain on |Σᵢⱼ|."""
function correlation_chain_order(Σ::Matrix{Float64})
    N = size(Σ, 1)
    used = falses(N)
    order = Int[]
    i = argmax(diag(Σ))
    used[i] = true
    push!(order, i)
    while length(order) < N
        last = order[end]
        best_j, best_s = 0, -Inf
        for j in 1:N
            if !used[j]
                s = abs(Σ[last, j])
                if s > best_s
                    best_s, best_j = s, j
                end
            end
        end
        used[best_j] = true
        push!(order, best_j)
    end
    return order
end

# ── Phase diagram: minimal χ for residual tolerance ─────────────────────────

"""
    dmrg_relative_residual(mps, mpo, E; H_norm) -> Float64

Report eq. (84) style: `‖(H−E)ψ‖₂ / ‖H‖₂`. `H_norm` defaults to `max(|E|, 1)`.
"""
function dmrg_relative_residual(
    mps::Vector{<:Array{Float64,3}}, mpo::Vector{<:Array{Float64,4}}, E::Real;
    H_norm::Union{Nothing,Real}=nothing,
)
    res = dmrg_residual(mps, mpo, E)
    denom = H_norm === nothing ? max(abs(E), 1.0) : Float64(H_norm)
    return res / denom
end

"""
    minimal_chi_dmrg(mpo; eps_res, chi_candidates, kwargs...) -> NamedTuple

Smallest `χ` in `chi_candidates` with relative residual ≤ `eps_res`.
Returns `(χ=..., E=..., residual=..., rel_residual=..., mps=...)`.
If none pass, returns the best (lowest rel_residual) candidate.
"""
function minimal_chi_dmrg(
    mpo::Vector{<:Array{Float64,4}};
    eps_res::Float64=1e-2,
    chi_candidates::Vector{Int}=[4, 8, 16, 24, 32, 48, 64],
    rng::AbstractRNG=Random.default_rng(),
    kwargs...
)
    best = nothing
    for χ in chi_candidates
        E, mps, _ = dmrg_ground_state(mpo; maxdim=χ, rng=rng, kwargs...)
        rel = dmrg_relative_residual(mps, mpo, E)
        cand = (; χ=χ, E=E, residual=dmrg_residual(mps, mpo, E), rel_residual=rel, mps=mps)
        if rel ≤ eps_res
            return cand
        end
        if best === nothing || rel < best.rel_residual
            best = cand
        end
    end
    return best
end

"""
    phase_diagram_point(N, K; kwargs...) -> NamedTuple

One §10.1 scaling point: random instance + minimal-χ search + metadata.
"""
function phase_diagram_point(
    N::Int, K::Int;
    factor_strength::Float64=1.0,
    Q::Int=2,
    eps_res::Float64=1e-2,
    chi_candidates::Vector{Int}=[4, 8, 16, 24, 32, 48, 64],
    rng::AbstractRNG=Random.default_rng(),
    dmrg_kwargs...
)
    model, Δ, B, cores = random_factor_model(N, K; Q=Q, factor_strength=factor_strength, rng=rng)
    result = minimal_chi_dmrg(cores; eps_res=eps_res, chi_candidates=chi_candidates, rng=rng, dmrg_kwargs...)
    return (;
        N, K, Q, factor_strength,
        hilbert_dim=hilbert_dim(model),
        chi=result.χ,
        E0=result.E,
        residual=result.residual,
        rel_residual=result.rel_residual,
        mps=result.mps,
        c_over_eta=sum(c_coeff(model) ./ model.η) / model.N,
        model, Δ, B, cores,
    )
end

# ── Baselines ─────────────────────────────────────────────────────────────────

"""
    product_diagonal_ground_state(model, Δ) -> (E0, ϕ0_dense)

Exact ground state when Σ = diag(Δ) only (Theorem 7.1).
"""
function product_diagonal_ground_state(model::MarketMakingModel, Δ::Vector{Float64})
    model_diag = MarketMakingModel(
        model.N, model.Qs, Matrix(Diagonal(Δ)), model.μ, model.k, model.γ, model.η,
    )
    H = build_hamiltonian_sparse(model_diag)
    vals, vecs = exact_ground_states(H; nev=1)
    return vals[1], vecs[1]
end

"""
    gaussian_harmonic_G(Σ, η) -> G

Precision matrix G from Proposition 7.4: G D_η G = c Σ with D_η = diag(η).
"""
function gaussian_harmonic_G(Σ::Matrix{Float64}, η::Vector{Float64}, c::Real)
    Dη = Diagonal(η)
    A = sqrt(c) * sqrt(Dη * Σ * Dη)
    G = Dη \ Matrix(A) / Dη
    G = 0.5 * (G + G')   # symmetrise numerically
    return G
end

"""
    gaussian_log_quote_ratio(q, qn, G) -> Float64

Theorem 7.4: log ϕ(q)/ϕ(qn) ≈ G(q−qn)·(q+qn)/2 style increment along one asset.
For single coordinate change q → q ± e_i:
  log ϕ(q)/ϕ(q+e_i) = (G q)_i + G_ii/2  (eq. 74).
"""
function gaussian_log_quote_ratio(q::Vector{Int}, i::Int, dir::Int, G::Matrix{Float64})
    ei = zeros(length(q))
    ei[i] = 1.0
    qf = Float64.(q)
    if dir == 1
        return dot(G * qf, ei) + G[i, i] / 2
    else
        return -dot(G * qf, ei) + G[i, i] / 2
    end
end

"""RMSE of Gaussian harmonic log-ratios vs exact φ₀ on `edges`."""
function gaussian_quote_rmse(model::MarketMakingModel, Δ::Vector{Float64}, ϕ_exact::Vector{Float64}, edges)
    G = gaussian_harmonic_G(model.Σ, model.η, c_coeff(model))
    Qs = model.Qs
    se, n = 0.0, 0
    maxabs = 0.0
    for e in edges
        qn = copy(e.q)
        qn[e.i] += e.dir
        i0 = config_to_index(e.q, Qs)
        i1 = config_to_index(qn, Qs)
        r_exact = log(ϕ_exact[i0]) - log(ϕ_exact[i1])
        r_gauss = gaussian_log_quote_ratio(e.q, e.i, e.dir, G)
        err = r_gauss - r_exact
        se += err^2
        maxabs = max(maxabs, abs(err))
        n += 1
    end
    return sqrt(se / n), maxabs
end

# ── Exact finite-horizon evolution (feasible for moderate |Q|) ───────────────

"""
    terminal_vector(model; ℓ) -> Vector{Float64}

Positive terminal vector v_q(T) = exp(−k ℓ(q)) on the full inventory grid.
"""
function terminal_vector(model::MarketMakingModel; ℓ::Function=q -> 0.0)
    Qs = model.Qs
    N = model.N
    D = hilbert_dim(model)
    v = Vector{Float64}(undef, D)
    @inbounds for idx in 1:D
        q = index_to_config(idx, Qs)
        v[idx] = exp(-model.k * ℓ(q))
    end
    return v
end

"""
    krylov_expmv(τ, H, v; krylovdim, tol) -> Vector{Float64}

Compute `exp(-τ H) v` for symmetric sparse `H` via Lanczos tridiagonalisation.
"""
function krylov_expmv(
    τ::Real, H::SparseMatrixCSC{Float64,Int}, v::Vector{Float64};
    krylovdim::Int=80, tol::Float64=1e-14,
)
    n = length(v)
    scale = norm(v)
    scale ≤ tol && return copy(v)
    vn = v ./ scale
    V = Matrix{Float64}(undef, n, krylovdim)
    α = Vector{Float64}(undef, krylovdim)
    β = Vector{Float64}(undef, krylovdim - 1)
    V[:, 1] = vn
    m_eff = krylovdim
    for j in 1:(krylovdim - 1)
        w = H * V[:, j]
        j > 1 && (w .-= β[j - 1] * V[:, j - 1])
        α[j] = dot(V[:, j], w)
        w .-= α[j] * V[:, j]
        βj = norm(w)
        if βj ≤ tol
            m_eff = j
            break
        end
        β[j] = βj
        V[:, j + 1] = w ./ βj
    end
    T = m_eff == 1 ? fill(α[1], 1, 1) :
        Matrix(SymTridiagonal(α[1:m_eff], β[1:m_eff - 1]))
    coeff = exp(-τ * T) * [1.0; zeros(m_eff - 1)]
    out = zeros(n)
    for k in 1:m_eff
        out .+= coeff[k] * V[:, k]
    end
    return out * scale
end

"""
    exact_imaginary_time_evolution(H, ψ0, τ_values) -> Vector{Vector{Float64}}

Exact ψ(τ) = exp(−τ H) ψ₀ for each τ in `τ_values` (Krylov matrix exponential).
Uses `ψ₀` normalised to unit ℓ² norm.
"""
function exact_imaginary_time_evolution(
    H::SparseMatrixCSC{Float64,Int}, ψ0::Vector{Float64}, τ_values::Vector{Float64};
    krylovdim::Int=80,
)
    ψ0n = ψ0 ./ norm(ψ0)
    out = Vector{Vector{Float64}}(undef, length(τ_values))
    for (k, τ) in pairs(τ_values)
        if τ == 0.0
            out[k] = copy(ψ0n)
        else
            out[k] = krylov_expmv(τ, H, ψ0n; krylovdim=krylovdim)
        end
        out[k] ./= norm(out[k])
    end
    return out
end

"""
    quote_rmse_vs_reference(ϕ, ϕ_ref, model, edges) -> (rmse, maxabs)
"""
function quote_rmse_vs_reference(ϕ, ϕ_ref, model::MarketMakingModel, edges)
    Qs = model.Qs
    ϕa = align_and_normalize(ϕ, ϕ_ref)
    return quote_log_ratio_errors(ϕa, ϕ_ref, Qs, edges)
end

"""
    finite_horizon_quote_convergence(model, H, ϕ0_ref, τ_values; edges, ℓ) -> DataFrame-like vectors

Track quote RMSE of ψ(τ) vs ground state ϕ₀ as τ grows (Theorem 4.6).
"""
function finite_horizon_quote_convergence(
    model::MarketMakingModel, H::SparseMatrixCSC{Float64,Int},
    ϕ0_ref::Vector{Float64}, τ_values::Vector{Float64};
    edges=nothing, ℓ::Function=q -> 0.0,
)
    edges === nothing && (edges = central_grid_edges(model.Qs))
    ψ0 = terminal_vector(model; ℓ=ℓ)
    ψτ = exact_imaginary_time_evolution(H, ψ0, τ_values)
    rmse = Float64[]
    maxe = Float64[]
    for ψ in ψτ
        r, m = quote_rmse_vs_reference(ψ, ϕ0_ref, model, edges)
        push!(rmse, r)
        push!(maxe, m)
    end
    return (; τ_values, rmse, maxe, ψτ)
end

# ── Doob chain (Theorem 4.8) ──────────────────────────────────────────────────

"""Factor `a = 1 + γ/k` (report eq. 31)."""
doob_a(model::MarketMakingModel) = 1.0 + model.γ / model.k

"""
    doob_rates_from_phi(ϕ, model) -> Vector{NamedTuple}

For each directed edge `q → q'` in the inventory grid, rate
`r(q→q') = a η_i ϕ(q')/ϕ(q)` when valid (Theorem 4.8).
"""
function doob_rates_from_phi(ϕ::Vector{Float64}, model::MarketMakingModel)
    Qs = model.Qs
    N = model.N
    a = doob_a(model)
    edges = NamedTuple{(:from,:to,:rate),Tuple{Int,Int,Float64}}[]
    D = hilbert_dim(model)
    for idx in 1:D
        q = index_to_config(idx, Qs)
        for i in 1:N
            if q[i] < Qs[i]
                qn = copy(q); qn[i] += 1
                j = config_to_index(qn, Qs)
                rate = a * model.η[i] * ϕ[j] / ϕ[idx]
                push!(edges, (; from=idx, to=j, rate=rate))
            end
            if q[i] > -Qs[i]
                qn = copy(q); qn[i] -= 1
                j = config_to_index(qn, Qs)
                rate = a * model.η[i] * ϕ[j] / ϕ[idx]
                push!(edges, (; from=idx, to=j, rate=rate))
            end
        end
    end
    return edges
end

"""Build sparse generator matrix from Doob rates (columns = outgoing from state)."""
function doob_generator(edges, D::Int)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    out_rates = zeros(Float64, D)
    for e in edges
        push!(rows, e.to); push!(cols, e.from); push!(vals, e.rate)
        out_rates[e.from] += e.rate
    end
    for i in 1:D
        push!(rows, i); push!(cols, i); push!(vals, -out_rates[i])
    end
    return sparse(rows, cols, vals, D, D)
end

"""
    simulate_doob_chain(edges, D, n_steps; rng, init) -> Vector{Int}

Discrete-time simulation with uniformization: each step picks a jump with
probability ∝ rate / total_rate.
"""
function simulate_doob_chain(
    edges, D::Int, n_steps::Int;
    rng::AbstractRNG=Random.default_rng(),
    init::Int=1,
)
    # adjacency lists
    adj = [Vector{NamedTuple{(:to,:rate),Tuple{Int,Float64}}}() for _ in 1:D]
    out_rates = zeros(Float64, D)
    for e in edges
        push!(adj[e.from], (; to=e.to, rate=e.rate))
        out_rates[e.from] += e.rate
    end
    state = init
    traj = Vector{Int}(undef, n_steps + 1)
    traj[1] = state
    for t in 1:n_steps
        rtot = out_rates[state]
        if rtot ≤ 0
            traj[t+1:end] .= state
            break
        end
        u = rand(rng) * rtot
        acc = 0.0
        for e in adj[state]
            acc += e.rate
            if u ≤ acc
                state = e.to
                break
            end
        end
        traj[t + 1] = state
    end
    return traj
end

"""Empirical occupancy vs π(q) ∝ ϕ(q)² on simulated trajectory."""
function doob_occupancy_error(traj::Vector{Int}, ϕ::Vector{Float64})
    D = length(ϕ)
    π = ϕ .^ 2
    π ./= sum(π)
    emp = zeros(Float64, D)
    for s in traj
        emp[s] += 1
    end
    emp ./= length(traj)
    return norm(emp - π)   # ℓ² distance
end

"""Estimate relaxation: time until empirical TV distance drops below threshold."""
function doob_mixing_steps(traj::Vector{Int}, ϕ::Vector{Float64}; threshold::Float64=0.05)
    D = length(ϕ)
    π = ϕ .^ 2
    π ./= sum(π)
    emp = zeros(Float64, D)
    for (t, s) in enumerate(traj)
        emp .*= (t - 1) / t
        emp[s] += 1 / t
        if norm(emp - π) ≤ threshold
            return t
        end
    end
    return length(traj)
end

# ── Factor-grid reduction baseline (report §7 factor-grid) ────────────────────

"""Per-factor inventory bounds `|f_k| ≤ Σ_i |B_{ik}| Q_i` over the box grid."""
function factor_loading_bounds(B::Matrix{Float64}, Qs::Vector{Int})
    N, K = size(B)
    return [ceil(Int, sum(abs(B[i, k]) * Qs[i] for i in 1:N)) for k in 1:K]
end

"""Map full inventory `q` to nearest factor-grid configuration."""
function inventory_to_factor_config(q::Vector{Int}, B::Matrix{Float64}, Qf::Vector{Int})
    f = B' * Float64.(q)
    return [clamp(round(Int, f[k]), -Qf[k], Qf[k]) for k in 1:length(Qf)]
end

"""
    factor_grid_reduced_model(model, Δ, B) -> (model_f, Qf, Δf, Bf)

Build a `K`-site factor-grid surrogate: latent inventory `f = B'q` with
`Σ_f = diag(Δ_f) + B_f B_f'` approximating the factor part of `Σ`.
"""
function factor_grid_reduced_model(
    model::MarketMakingModel, Δ::Vector{Float64}, B::Matrix{Float64},
)
    Qf = max.(1, factor_loading_bounds(B, model.Qs))
    K = length(Qf)
    Δf = [sum(Δ[i] * B[i, k]^2 for i in 1:model.N) for k in 1:K]
    Bf = Matrix{Float64}(I, K, K)   # Σ_f = diag(Δf) + I
    ηf = [sum(model.η[i] * abs(B[i, k]) for i in 1:model.N) / max(sum(abs, B[:, k]), 1e-12)
          for k in 1:K]
    mf = MarketMakingModel(K, Qf, Matrix(Diagonal(Δf) + Bf * Bf'), zeros(K),
                           model.k, model.γ, ηf)
    return mf, Qf, Δf, Bf
end

"""
    factor_grid_ground_state(model, Δ, B) -> (E0, ϕ_f, Qf, model_f)

Exact diagonalization on the factor grid when `|Q_f| ≤ max_exact`, else DMRG.
"""
function factor_grid_ground_state(
    model::MarketMakingModel, Δ::Vector{Float64}, B::Matrix{Float64};
    max_exact::Int=20_000, dmrg_maxdim::Int=32, dmrg_sweeps::Int=8,
    rng::AbstractRNG=Random.default_rng(),
)
    mf, Qf, Δf, Bf = factor_grid_reduced_model(model, Δ, B)
    if hilbert_dim(mf) ≤ max_exact
        H = build_hamiltonian_sparse(mf)
        vals, vecs = exact_ground_states(H; nev=1, rng=rng)
        return vals[1], vecs[1], Qf, mf
    end
    cores = build_mpo_cores(mf, Δf, Bf)
    E, mps, _ = dmrg_ground_state(cores; maxdim=dmrg_maxdim, n_sweeps=dmrg_sweeps, rng=rng)
    ϕ = mps_to_lex_vector(mps, site_dims(mf))
    return E, ϕ, Qf, mf
end

"""Lift factor-grid amplitudes to the full inventory configuration `q`."""
function factor_grid_amplitude(q::Vector{Int}, ϕ_f::Vector{Float64}, B::Matrix{Float64}, Qf::Vector{Int})
    fk = inventory_to_factor_config(q, B, Qf)
    return ϕ_f[config_to_index(fk, Qf)]
end

"""RMSE of factor-grid log-ratios vs exact `φ₀` on `edges`."""
function factor_grid_quote_rmse(
    model::MarketMakingModel, Δ::Vector{Float64}, B::Matrix{Float64},
    ϕ_exact::Vector{Float64}, edges;
    kwargs...
)
    _, ϕ_f, Qf, _ = factor_grid_ground_state(model, Δ, B; kwargs...)
    Qs = model.Qs
    se, n, maxabs = 0.0, 0, 0.0
    for e in edges
        qn = copy(e.q)
        qn[e.i] += e.dir
        i0 = config_to_index(e.q, Qs)
        i1 = config_to_index(qn, Qs)
        ϕ0 = factor_grid_amplitude(e.q, ϕ_f, B, Qf)
        ϕ1 = factor_grid_amplitude(qn, ϕ_f, B, Qf)
        r_exact = log(ϕ_exact[i0]) - log(ϕ_exact[i1])
        r_fg = log(ϕ0) - log(ϕ1)
        err = r_fg - r_exact
        se += err^2
        maxabs = max(maxabs, abs(err))
        n += 1
    end
    return sqrt(se / n), maxabs, ϕ_f, Qf
end

# ── Baseline comparison at equal quote error ──────────────────────────────────

"""
    baseline_quote_comparison(model, Δ, B, ϕ0, edges) -> NamedTuple

Quote RMSE for product, Gaussian, and factor-grid baselines vs exact `φ₀`.
"""
function baseline_quote_comparison(
    model::MarketMakingModel, Δ::Vector{Float64}, B::Matrix{Float64},
    ϕ0::Vector{Float64}, edges;
    kwargs...
)
    _, ϕ_diag = product_diagonal_ground_state(model, Δ)
    rmse_prod, max_prod = quote_log_ratio_errors(
        align_and_normalize(ϕ_diag, ϕ0), ϕ0, model.Qs, edges)
    rmse_g, max_g = gaussian_quote_rmse(model, Δ, ϕ0, edges)
    rmse_fg, max_fg, _, _ = factor_grid_quote_rmse(model, Δ, B, ϕ0, edges; kwargs...)
    return (; product=(rmse=rmse_prod, max=max_prod),
            gaussian=(rmse=rmse_g, max=max_g),
            factor_grid=(rmse=rmse_fg, max=max_fg))
end

# ── MPS finite-horizon convergence ────────────────────────────────────────────

"""
    finite_horizon_mps_convergence(model, mpo, H, ϕ0_ref, τ_values; kwargs...)
        -> (; τ_values, rmse, maxe, mps_snapshots)

MPS split-step imaginary-time evolution; quote RMSE vs exact ground state.
"""
function finite_horizon_mps_convergence(
    model::MarketMakingModel, mpo::Vector{<:Array{Float64,4}},
    ϕ0_ref::Vector{Float64}, τ_values::Vector{Float64};
    edges=nothing, ℓ::Function=q -> 0.0,
    dt::Float64=0.1, maxdim::Int=32, kwargs...
)
    edges === nothing && (edges = central_grid_edges(model.Qs))
    ψ0 = terminal_vector(model; ℓ=ℓ)
    mps_hist = mps_imaginary_time_evolution(mpo, model, ψ0; τ_values=τ_values,
                                            dt=dt, maxdim=maxdim, kwargs...)
    rmse = Float64[]
    maxe = Float64[]
    for mps in mps_hist
        ψ = mps_to_dense_lex(mps, model)
        r, m = quote_rmse_vs_reference(ψ, ϕ0_ref, model, edges)
        push!(rmse, r)
        push!(maxe, m)
    end
    return (; τ_values, rmse, maxe, mps_snapshots=mps_hist)
end

# ── Phase-diagram grid sweep ──────────────────────────────────────────────────

"""
    phase_diagram_sweep(specs; eps_res, chi_candidates, rng, dmrg_kwargs...) -> Vector

Run `phase_diagram_point` for each `(N, K, Q, factor_strength)` tuple in `specs`.
"""
function phase_diagram_sweep(
    specs::Vector{<:Tuple};
    eps_res::Float64=1e-2,
    chi_candidates::Vector{Int}=[4, 8, 16, 24, 32, 48, 64, 96, 128],
    seed_base::Int=0,
    dmrg_kwargs...
)
    results = NamedTuple[]
    for (i, spec) in enumerate(specs)
        N, K, Q, fs = spec
        rng = MersenneTwister(seed_base + i)
        pt = phase_diagram_point(N, K; Q=Q, factor_strength=fs, eps_res=eps_res,
            chi_candidates=chi_candidates, rng=rng, dmrg_kwargs...)
        push!(results, (; pt..., seed=seed_base + i, d=2Q + 1))
    end
    return results
end

"""Default §10.1 phase-diagram grid: `N ∈ {20,50,100}`, `K ∈ {1,2,3,5}`, `d ∈ {5,11}`."""
function default_phase_diagram_specs()
    specs = Tuple{Int,Int,Int,Float64}[]
    for N in (20, 50, 100)
        for K in (1, 2, 3, 5)
            for Q in (2, 5)   # d = 2Q+1 ∈ {5, 11}
                fs = Q == 2 ? 1.0 : 0.8
                push!(specs, (N, K, Q, fs))
            end
        end
    end
    return specs
end

export report_benchmark_model
export random_factor_model, permute_factor_model, seriation_order, correlation_chain_order
export dmrg_relative_residual, minimal_chi_dmrg, phase_diagram_point
export product_diagonal_ground_state, gaussian_harmonic_G, gaussian_log_quote_ratio
export gaussian_quote_rmse
export krylov_expmv, terminal_vector, exact_imaginary_time_evolution
export quote_rmse_vs_reference, finite_horizon_quote_convergence
export doob_a, doob_rates_from_phi, doob_generator
export simulate_doob_chain, doob_occupancy_error, doob_mixing_steps
export factor_loading_bounds, factor_grid_reduced_model, factor_grid_ground_state
export factor_grid_amplitude, factor_grid_quote_rmse, baseline_quote_comparison
export finite_horizon_mps_convergence
export phase_diagram_sweep, default_phase_diagram_specs
