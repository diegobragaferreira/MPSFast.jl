# market_making_imtime.jl — MPS imaginary-time evolution (split-step TEBD) for
# ψ(τ) = exp(−τ H) ψ₀ without forming the full Hilbert-space vector.
# Reuses two-site MPO environments from dmrg.jl (Coutinho §10.1, finite horizon).

using TensorOperations
using LinearAlgebra
using Random

"""
    krylov_expmv_op(τ, f, v0; krylovdim, tol) -> Vector

Matrix-free `exp(−τ A) v₀` for symmetric linear operator `f(v)`.
"""
function krylov_expmv_op(
    f, v0::Vector{Float64};
    τ::Real, krylovdim::Int=50, tol::Float64=1e-12,
)
    scale = norm(v0)
    scale ≤ tol && return copy(v0)
    vn = v0 ./ scale
    n = length(v0)
    V = Matrix{Float64}(undef, n, krylovdim)
    α = Vector{Float64}(undef, krylovdim)
    β = Vector{Float64}(undef, krylovdim - 1)
    V[:, 1] = vn
    m_eff = krylovdim
    for j in 1:(krylovdim - 1)
        w = f(V[:, j])
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

"""Apply `exp(−Δτ H_eff)` to a merged two-site tensor."""
function expm_heff_twosite(
    Lenv::Array{Float64,3}, W1::Array{Float64,4}, W2::Array{Float64,4},
    Renv::Array{Float64,3}, psi::Array{Float64,4}, Δτ::Real;
    krylovdim::Int=40, dense_limit::Int=1024,
)
    sz = size(psi)
    D = length(psi)
    f = v -> vec(apply_Heff_twosite(Lenv, W1, W2, Renv, reshape(v, sz)))
    if D ≤ dense_limit
        Heff = zeros(D, D)
        ej = zeros(D)
        @inbounds for col in 1:D
            ej[col] = 1.0
            Heff[:, col] = f(ej)
            ej[col] = 0.0
        end
        Heff = 0.5 * (Heff + Heff')
        return reshape(exp(-Δτ * Heff) * vec(psi), sz)
    end
    return reshape(krylov_expmv_op(f, vec(psi); τ=Δτ, krylovdim=min(krylovdim, D)), sz)
end

"""
    imtime_sweep!(mps, mpo, Δτ; maxdim, cutoff, krylovdim) -> mps

One first-order Trotter sweep: at each bond apply `exp(−Δτ H_eff)` and truncate.
"""
function imtime_sweep!(
    mps::Vector{<:Array{Float64,3}}, mpo::Vector{<:Array{Float64,4}}, Δτ::Real;
    maxdim::Int=32, cutoff::Float64=1e-12, krylovdim::Int=40,
)
    N = length(mps)
    right_canonicalize_mps!(mps)
    Lenv, Renv = mpo_environments(mps, mpo)
    for j in 1:(N - 1)
        psi = merge_two_site(mps[j], mps[j + 1])
        psi_new = expm_heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], psi, Δτ;
                                    krylovdim=krylovdim)
        A1, A2, _ = split_two_site(psi_new, maxdim, :right; cutoff=cutoff)
        mps[j], mps[j + 1] = A1, A2
        Lenv[j + 1] = _update_Lenv(Lenv[j], mps[j], mpo[j])
    end
    for j in (N - 1):-1:1
        psi = merge_two_site(mps[j], mps[j + 1])
        psi_new = expm_heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], psi, Δτ;
                                    krylovdim=krylovdim)
        A1, A2, _ = split_two_site(psi_new, maxdim, :left; cutoff=cutoff)
        mps[j], mps[j + 1] = A1, A2
        Renv[j + 2] = _update_Renv(Renv[j + 3], mps[j + 1], mpo[j + 1])
    end
    nrm = sqrt(mps_inner_product(mps, mps))
    nrm > 0 && (mps[1] ./= nrm)
    return mps
end

"""
    dense_to_mps(model, ψ; maxdim) -> mps

TT-SVD compress a dense lexicographic state vector to MPS.
"""
function dense_to_mps(model::MarketMakingModel, ψ::Vector{Float64}; maxdim::Int=32)
    dims = site_dims(model)
    tens = lex_vector_to_tensor(ψ, dims)
    cores, _ = tt_svd(tens; maxdim=maxdim)
    return [Array{Float64,3}(c) for c in cores]
end

"""
    mps_imaginary_time_evolution(mpo, model, ψ0; τ_values, dt, maxdim, kwargs...)
        -> Vector{mps}

Split-step TEBD imaginary-time evolution from initial dense vector `ψ0`
(lexicographic). Returns normalised MPS snapshots at each target `τ`.
"""
function mps_imaginary_time_evolution(
    mpo::Vector{<:Array{Float64,4}}, model::MarketMakingModel, ψ0::Vector{Float64};
    τ_values::Vector{Float64}=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    dt::Float64=0.1,
    maxdim::Int=32,
    kwargs...
)
    mps = dense_to_mps(model, ψ0; maxdim=maxdim)
    nrm = norm(ψ0)
    nrm > 0 && (mps[1] .*= (nrm / sqrt(mps_inner_product(mps, mps))))
    out = Vector{Vector{Array{Float64,3}}}(undef, length(τ_values))
    τ_cur = 0.0
    idx = 1
    if τ_values[1] == 0.0
        out[1] = deepcopy(mps)
        idx = 2
    end
    while idx <= length(τ_values)
        τ_tgt = τ_values[idx]
        while τ_cur < τ_tgt - 1e-14
            step = min(dt, τ_tgt - τ_cur)
            imtime_sweep!(mps, mpo, step; maxdim=maxdim, kwargs...)
            τ_cur += step
        end
        out[idx] = deepcopy(mps)
        idx += 1
    end
    return out
end

"""Lexicographic dense vector from MPS (for quote metrics)."""
function mps_to_dense_lex(mps::Vector{<:Array{Float64,3}}, model::MarketMakingModel)
    return mps_to_lex_vector([Array(a) for a in mps], site_dims(model))
end

export krylov_expmv_op, expm_heff_twosite, imtime_sweep!
export dense_to_mps, mps_imaginary_time_evolution, mps_to_dense_lex
