# market_making_bounds.jl — Model-specific entanglement / bond-dimension bounds
# for the Coutinho market-making Hamiltonian.
#
# Two complementary routes (see notes §7):
#   1. Factor-coupling control: Σ = Δ + BBᵀ ⇒ bound entropy via fidelity to the
#      diagonal-Σ product ground state (Theorem 7.1) + Fannes inequality.
#   2. Gapped 1D area law: spectral gap Δ_H ⇒ correlation length ξ ≲ 1/Δ_H and
#      S_max ≲ O(log d / Δ_H) (Hastings-style; validated empirically).

using LinearAlgebra
using Random

max_site_dim(model::MarketMakingModel) = maximum(site_dims(model))

"""Dimensionless factor-coupling strength κ = ‖B‖_F² / λ_min(Δ)."""
function factor_coupling_strength(Δ::AbstractVector{<:Real}, B::AbstractMatrix{<:Real})
    λmin = minimum(Δ)
    λmin = max(λmin, 1e-14)
    return norm(B)^2 / λmin
end

"""Largest left/right Hilbert factor across all bipartite cuts."""
function max_cut_hilbert_dim(model::MarketMakingModel)
    dims = site_dims(model)
    N = length(dims)
    N <= 1 && return dims[1]
    left = 1
    right = prod(dims)
    m = 0
    @inbounds for ℓ in 1:(N - 1)
        left *= dims[ℓ]
        right ÷= dims[ℓ]
        m = max(m, left, right)
    end
    return m
end

"""Max over cuts of `min(dim_left, dim_right)` — tightest log-dimension scale per cut."""
function max_symmetric_cut_dim(model::MarketMakingModel)
    dims = site_dims(model)
    N = length(dims)
    N <= 1 && return dims[1]
    left = 1
    right = prod(dims)
    m = 0
    @inbounds for ℓ in 1:(N - 1)
        left *= dims[ℓ]
        right ÷= dims[ℓ]
        m = max(m, min(left, right))
    end
    return m
end

"""
    hamiltonian_spectral_gap(model; nev=2, rng) -> (E0, E1, gap)

Smallest two eigenvalues of `H` via sparse Lanczos.
"""
function hamiltonian_spectral_gap(
    model::MarketMakingModel;
    nev::Int=2,
    rng::AbstractRNG=Random.default_rng(),
)
    H = build_hamiltonian_sparse(model)
    vals, _ = exact_ground_states(H; nev=nev, rng=rng)
    return vals[1], vals[2], vals[2] - vals[1]
end

"""
    exact_ground_state_vector(model; rng) -> (E0, phi)

Ground-state amplitude in lexicographic convention.
"""
function exact_ground_state_vector(
    model::MarketMakingModel;
    rng::AbstractRNG=Random.default_rng(),
)
    H = build_hamiltonian_sparse(model)
    vals, vecs = exact_ground_states(H; nev=1, rng=rng)
    return vals[1], vecs[1]
end

# ── Inventory moments from φ₀ ─────────────────────────────────────────────────

"""Probability weights `p(q) ∝ |φ(q)|²` (normalised)."""
function ground_state_probabilities(phi::AbstractVector{<:Real})
    p = abs2.(phi)
    z = sum(p)
    z > 0 && (p ./= z)
    return p
end

"""
    inventory_moments(phi, model) -> (mean_q, cov_q)

Exact expectations under `p(q) ∝ |φ(q)|²` on the inventory grid.
"""
function inventory_moments(phi::AbstractVector{<:Real}, model::MarketMakingModel)
    N, Qs = model.N, model.Qs
    p = ground_state_probabilities(phi)
    mean_q = zeros(Float64, N)
    eq = zeros(Float64, N)
    eq2 = zeros(Float64, N, N)
    D = length(p)
    @inbounds for lin in 1:D
        w = p[lin]
        iszero(w) && continue
        q = index_to_config(lin, Qs)
        for i in 1:N
            qi = q[i]
            mean_q[i] += w * qi
            eq[i] += w * qi
            for j in 1:N
                eq2[i, j] += w * qi * q[j]
            end
        end
    end
    cov_q = eq2 - mean_q * mean_q'
    return mean_q, cov_q
end

"""Connected inventory correlations `C_ij = Cov(q_i, q_j)`."""
function connected_inventory_correlations(phi::AbstractVector{<:Real}, model::MarketMakingModel)
    _, cov_q = inventory_moments(phi, model)
    return cov_q
end

"""
    separation_correlation_profile(phi, model) -> Vector{Float64}

For each separation `d = 1,…,N−1`, the maximum absolute normalised connected
correlation `|C_ij| / √(Var_i Var_j)` over pairs with `|i−j| = d`.
"""
function separation_correlation_profile(phi::AbstractVector{<:Real}, model::MarketMakingModel)
    N = model.N
    C = connected_inventory_correlations(phi, model)
    var_q = max.(diag(C), 1e-30)
    prof = zeros(Float64, max(N - 1, 0))
    for d in 1:(N - 1)
        m = 0.0
        for i in 1:(N - d)
            j = i + d
            denom = sqrt(var_q[i] * var_q[j])
            m = max(m, abs(C[i, j]) / denom)
        end
        prof[d] = m
    end
    return prof
end

"""
    correlation_length_estimate(profile) -> (xi, decay_rate)

Fit `log profile[d] ≈ log A − d/ξ` for `d ≥ 1` with `profile[d] > 0`.
Returns `(ξ, 1/ξ)`; `xi = Inf` if no decay is detected.
"""
function correlation_length_estimate(profile::AbstractVector{<:Real})
    ds = Int[]
    ys = Float64[]
    for d in eachindex(profile)
        v = profile[d]
        if v > 1e-14
            push!(ds, d)
            push!(ys, log(v))
        end
    end
    isempty(ds) && return (Inf, 0.0)
    length(ds) == 1 && return (Inf, 0.0)
    X = hcat(ones(length(ds)), Float64.(ds))
    coef = X \ ys
    slope = coef[2]
    slope >= 0 && return (Inf, 0.0)
    xi = -1.0 / slope
    return (xi, -slope)
end

# ── Entropy from exact ground state ───────────────────────────────────────────

"""Maximum bipartite von Neumann entropy (nats) across all cuts."""
function max_bipartite_entropy(phi::AbstractVector{<:Real}, model::MarketMakingModel)
    dims = site_dims(model)
    tens = lex_vector_to_tensor(phi, dims)
    entr = bipartite_entropies_exact(tens)
    return maximum(entr)
end

# ── Route 1: factor coupling + Fannes (diagonal reference) ────────────────────

"""Binary entropy `h₂(t) = −t log t − (1−t) log(1−t)` for `t ∈ [0,1]`."""
function binary_entropy(t::Real)
    t = clamp(Float64(t), 1e-30, 1 - 1e-30)
    return -t * log(t) - (1 - t) * log(1 - t)
end

"""
    diagonal_product_ground_state(model, Δ) -> phi_diag

Exact product ground state when `Σ = diag(Δ)` (Theorem 7.1).
Each site solves its own 1D stoquastic Hamiltonian
`h_i = c Δ_i q_i² − k μ_i q_i − η_i(L⁺_i + L⁻_i)`.
"""
function diagonal_product_ground_state(model::MarketMakingModel, Δ::AbstractVector{<:Real})
    N, Qs, μ, k = model.N, model.Qs, model.μ, model.k
    c = c_coeff(model)
    dims = site_dims(model)
    local_vecs = Vector{Vector{Float64}}(undef, N)
    @inbounds for i in 1:N
        d = dims[i]
        Lp, Lm = local_shift_ops(Qs[i])
        qi = Diagonal(Float64.(-Qs[i]:Qs[i]))
        hi = Matrix(c * Δ[i] * qi^2 .- k * μ[i] * qi .- model.η[i] * (Lp + Lm))
        vals, vecs = exact_ground_states(hi; nev=1)
        v = vecs[1]
        v ./= norm(v)
        v[argmax(abs.(v))] < 0 && (v .*= -1)
        local_vecs[i] = v
    end
    D = prod(dims)
    phi = Vector{Float64}(undef, D)
    @inbounds for lin in 1:D
        q = index_to_config(lin, Qs)
        amp = 1.0
        for i in 1:N
            σ = q[i] + Qs[i] + 1
            amp *= local_vecs[i][σ]
        end
        phi[lin] = amp
    end
    phi ./= norm(phi)
    j = argmax(abs.(phi))
    phi[j] < 0 && (phi .*= -1)
    return phi
end

"""Squared overlap `|⟨φ_a, φ_b⟩|²` between two normalised state vectors."""
function state_fidelity(phi_a::AbstractVector{<:Real}, phi_b::AbstractVector{<:Real})
    return abs2(dot(phi_a, phi_b))
end

"""
    fannes_entropy_bound(fidelity, d_eff) -> Float64

Fannes–Audenaert: if a bipartite reduced state has fidelity `F` to a separable
reference with zero entropy, then its entropy is at most
`t log(d_eff−1) + h₂(t)` with `t = √(1−F)`.
Use `d_eff = max_cut_hilbert_dim(model)` (not the full `|Q|`).
"""
function fannes_entropy_bound(fidelity::Real, d_eff::Int)
    d_eff = max(d_eff, 2)
    F = clamp(Float64(fidelity), 0.0, 1.0)
    t = sqrt(max(1.0 - F, 0.0))
    t <= 1e-15 && return 0.0
    return t * log(d_eff - 1) + binary_entropy(t)
end

"""
    factor_coupling_fannes_bound(model, Δ, B, phi) -> NamedTuple

Route 1: fidelity of `phi` to the diagonal-Σ product reference ⇒ Fannes upper bound on `S_max`.
"""
function factor_coupling_fannes_bound(
    model::MarketMakingModel,
    Δ::AbstractVector{<:Real},
    B::AbstractMatrix{<:Real},
    phi::AbstractVector{<:Real},
)
    phi_diag = diagonal_product_ground_state(model, Δ)
    F = state_fidelity(phi, phi_diag)
    d_cut = max_symmetric_cut_dim(model)
    S_bound = fannes_entropy_bound(F, d_cut)
    κ = factor_coupling_strength(Δ, B)
    return (; κ, fidelity=F, d_cut, S_fannes=S_bound)
end

"""
    factor_coupling_perturbation_bound(κ, N, d_max) -> Float64

Heuristic second-order bound for weak factor coupling:
`S_max ≲ N κ² log(d_max) / (1 + κ)` (validated empirically; not rigorous alone).
"""
function factor_coupling_perturbation_bound(κ::Real, N::Int, d_max::Int)
    κ = Float64(κ)
    return N * κ^2 * log(Float64(d_max)) / (1.0 + κ)
end

# ── Route 2: gapped area law + correlation length ─────────────────────────────

"""
    hastings_entropy_bound(gap, d_max; prefactor=6.0) -> Float64

Standard 1D gapped area-law scale: `S_max ≲ prefactor · log(d_max) / gap`.
`prefactor` is calibrated on the report §8 instance (default 6).
"""
function hastings_entropy_bound(gap::Real, d_max::Int; prefactor::Real=6.0)
    gap = max(Float64(gap), 1e-14)
    return Float64(prefactor) * log(Float64(d_max)) / gap
end

"""
    correlation_length_gap_bound(gap; prefactor=1.5) -> Float64

Empirical scale `ξ ≲ prefactor / gap` (calibrated on §8 benchmark).
"""
function correlation_length_gap_bound(gap::Real; prefactor::Real=1.5)
    gap = max(Float64(gap), 1e-14)
    return Float64(prefactor) / gap
end

"""
    area_law_from_correlation_length(xi, d_max) -> Float64

Entropy scale `S_max ≲ ξ · log(d_max)` when correlations decay on scale `ξ`.
"""
function area_law_from_correlation_length(xi::Real, d_max::Int)
    isinf(xi) || xi <= 0 && return Inf
    return Float64(xi) * log(Float64(d_max))
end

"""Conservative MPS bond dimension `⌈exp(S)⌉` from entropy bound."""
chi_from_entropy_bound(S::Real) = ceil(Int, exp(Float64(S)))

# ── Full audit ────────────────────────────────────────────────────────────────

"""
    entropy_bounds_audit(model, Δ, B; phi=nothing, rng) -> NamedTuple

Compute measured `S_max`, correlation length `ξ`, spectral gap, and compare to
both bound routes. If `phi` is omitted, the exact ground state is computed.
"""
function entropy_bounds_audit(
    model::MarketMakingModel,
    Δ::AbstractVector{<:Real},
    B::AbstractMatrix{<:Real};
    phi::Union{Nothing,AbstractVector{<:Real}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    if phi === nothing
        _, phi = exact_ground_state_vector(model; rng=rng)
    end
    E0, _, gap = hamiltonian_spectral_gap(model; rng=rng)
    d_max = max_site_dim(model)
    d_cut = max_symmetric_cut_dim(model)
    S_meas = max_bipartite_entropy(phi, model)
    prof = separation_correlation_profile(phi, model)
    ξ, decay = correlation_length_estimate(prof)
    route1 = factor_coupling_fannes_bound(model, Δ, B, phi)
    S_pert = factor_coupling_perturbation_bound(route1.κ, model.N, d_max)
    S_hastings = hastings_entropy_bound(gap, d_max)
    ξ_gap = correlation_length_gap_bound(gap)
    S_xi = area_law_from_correlation_length(ξ, d_max)
    S_xi_gap = area_law_from_correlation_length(ξ_gap, d_max)
    S_combined = min(route1.S_fannes, S_hastings, S_pert, S_xi_gap)
    return (
        E0=E0,
        gap=gap,
        d_max=d_max,
        d_cut=d_cut,
        S_measured=S_meas,
        correlation_length=ξ,
        correlation_decay=decay,
        κ=route1.κ,
        fidelity_diagonal=route1.fidelity,
        S_fannes=route1.S_fannes,
        S_perturbation=S_pert,
        S_hastings=S_hastings,
        xi_gap_bound=ξ_gap,
        S_from_xi=S_xi,
        S_from_xi_gap=S_xi_gap,
        S_combined=S_combined,
        chi_measured=chi_from_entropy_bound(S_meas),
        chi_fannes=chi_from_entropy_bound(route1.S_fannes),
        chi_hastings=chi_from_entropy_bound(S_hastings),
        chi_combined=chi_from_entropy_bound(S_combined),
        profile=prof,
    )
end

"""
    entropy_bounds_sweep(specs; rng) -> Vector{NamedTuple}

Run `entropy_bounds_audit` on a vector of `(model, Δ, B)` triples or NamedTuples.
"""
function entropy_bounds_sweep(
    specs;
    rng::AbstractRNG=Random.default_rng(),
)
    out = NamedTuple[]
    for spec in specs
        model = hasproperty(spec, :model) ? spec.model : spec[1]
        Δ = hasproperty(spec, :Δ) ? spec.Δ : spec[2]
        B = hasproperty(spec, :B) ? spec.B : spec[3]
        push!(out, entropy_bounds_audit(model, Δ, B; rng=rng))
    end
    return out
end

export factor_coupling_strength, max_site_dim, max_cut_hilbert_dim, max_symmetric_cut_dim, hamiltonian_spectral_gap
export exact_ground_state_vector, ground_state_probabilities, inventory_moments
export connected_inventory_correlations, separation_correlation_profile
export correlation_length_estimate, max_bipartite_entropy
export diagonal_product_ground_state, state_fidelity, fannes_entropy_bound
export factor_coupling_fannes_bound, factor_coupling_perturbation_bound
export hastings_entropy_bound, correlation_length_gap_bound
export area_law_from_correlation_length, chi_from_entropy_bound
export entropy_bounds_audit, entropy_bounds_sweep
