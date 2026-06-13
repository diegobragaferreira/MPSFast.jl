"""
    MPSFast.Encoders

Path encoders that translate continuous-valued financial paths to integer
physical-leg indices (and optional feature matrices Φ) consumed by the
MPSFast training and sampling kernels.

# Exported types

    PathEncoder               — abstract interface
    BasisEncoder(m)           — one-hot on 2^m uniform buckets (1 site/timestep)
    BinaryEncoder(m)          — bit expansion, d=2 per site (m sites/timestep)
    TrigEncoder(m, d_feat)    — trig Fourier features, activates Gram training

# Exported functions

    chain_length(enc, M)      — MPS chain length for M Heston timesteps
    site_dim(enc)             — physical leg dimension d
    encode_paths(enc, S)      — (N,M) Real matrix → (N,Ml) Int matrix
    decode_paths(enc, xi)     — inverse of encode_paths
    fit_grid!(enc, S)         — calibrate Smin/Smax to training data once
    feature_map(enc)          — nothing or K×d Φ matrix (TrigEncoder only)
    sample_paths(enc, mps, n) — sample n paths and decode to real values
    encode_labeled_paths(enc, S, y)          — path + label site for classification
    classification_chain_length(enc, M)      — path sites + 1 label site
    label_site_dim(n_classes)                — physical dim for label site
    encoder_summary(enc, M, D_max)           — pretty-print encoder config
"""
module Encoders

using Random
using Distributions
using LinearAlgebra
using Base.Threads

import ..sample_paths_feature_map

export PathEncoder
export BasisEncoder, BinaryEncoder, TrigEncoder
export chain_length, site_dim, encode_paths, decode_paths, fit_grid!, feature_map
export sample_paths
export encode_labeled_paths, classification_chain_length, label_site_dim
export encoder_summary

# ─── Abstract interface ───────────────────────────────────────────────────────

"""
    PathEncoder

Abstract type for all path encoders. Subtypes must implement:
* `chain_length(enc, M)` — MPS chain length for `M` Heston timesteps.
* `site_dim(enc)`        — physical leg dimension `d`.
* `encode_paths(enc, S)` — `Matrix{<:Real}` (N, M) → `Matrix{Int}` (N, Ml).
* `decode_paths(enc, xi)`— inverse of `encode_paths`.
* `fit_grid!(enc, S)`    — calibrate support to data (sets Smin/Smax).
"""
abstract type PathEncoder end

chain_length(::PathEncoder, ::Int)           = error("PathEncoder subtype must implement chain_length")
site_dim(::PathEncoder)                      = error("PathEncoder subtype must implement site_dim")
encode_paths(::PathEncoder, ::AbstractMatrix)  = error("PathEncoder subtype must implement encode_paths")
decode_paths(::PathEncoder, ::AbstractMatrix)  = error("PathEncoder subtype must implement decode_paths")
fit_grid!(::PathEncoder, ::AbstractMatrix)     = error("PathEncoder subtype must implement fit_grid!")

"""
    feature_map(enc) -> nothing or K×d Matrix

Return `nothing` for one-hot encoders (BasisEncoder, BinaryEncoder).
For `TrigEncoder`, return a `K × d_features` matrix of trig harmonics that
activates Gram-weighted transfers inside the training kernel.
"""
feature_map(::PathEncoder) = nothing

# ─── BasisEncoder ─────────────────────────────────────────────────────────────

"""
    BasisEncoder(m; Smin = NaN, Smax = NaN)

One-hot bucketing on a uniform `[Smin, Smax]` grid of `d = 2^m` levels.
MPS chain length equals `M` (one site per timestep).

Call `fit_grid!(enc, paths)` before `encode_paths` if `Smin`/`Smax` are not set.
"""
mutable struct BasisEncoder <: PathEncoder
    m::Int
    Smin::Float64
    Smax::Float64
end

BasisEncoder(m::Int; Smin::Real = NaN, Smax::Real = NaN) =
    BasisEncoder(m, Float64(Smin), Float64(Smax))

chain_length(::BasisEncoder, M::Int) = M
site_dim(enc::BasisEncoder)          = 2^enc.m

function fit_grid!(enc::BasisEncoder, paths::AbstractMatrix)
    enc.Smin = Float64(minimum(paths))
    enc.Smax = Float64(maximum(paths))
    return enc
end

function encode_paths(enc::BasisEncoder, paths::AbstractMatrix)
    @assert isfinite(enc.Smin) && isfinite(enc.Smax) "Call fit_grid!(enc, paths) first."
    d  = site_dim(enc)
    xi = clamp.(floor.(Int, (d - 1) .* (paths .- enc.Smin) ./ (enc.Smax - enc.Smin)) .+ 1, 1, d)
    return xi
end

function decode_paths(enc::BasisEncoder, xi::AbstractMatrix)
    d = site_dim(enc)
    return enc.Smin .+ ((xi .- 1) ./ (d - 1)) .* (enc.Smax - enc.Smin)
end

# ─── BinaryEncoder ────────────────────────────────────────────────────────────

"""
    BinaryEncoder(m; Smin = NaN, Smax = NaN)

Binary expansion: each timestep is bucketed into `2^m` levels and split into
`m` bits (MSB first), one per MPS site. Physical dimension is `d = 2`; chain
length is `M·m`. Per-pair-update cost is `O(D³)`, independent of `m`.
"""
mutable struct BinaryEncoder <: PathEncoder
    m::Int
    Smin::Float64
    Smax::Float64
end

BinaryEncoder(m::Int; Smin::Real = NaN, Smax::Real = NaN) =
    BinaryEncoder(m, Float64(Smin), Float64(Smax))

chain_length(enc::BinaryEncoder, M::Int) = M * enc.m
site_dim(::BinaryEncoder)                = 2

function fit_grid!(enc::BinaryEncoder, paths::AbstractMatrix)
    enc.Smin = Float64(minimum(paths))
    enc.Smax = Float64(maximum(paths))
    return enc
end

function encode_paths(enc::BinaryEncoder, paths::AbstractMatrix)
    @assert isfinite(enc.Smin) && isfinite(enc.Smax) "Call fit_grid!(enc, paths) first."
    N, M     = size(paths)
    m        = enc.m
    d_levels = 2^m
    buckets  = clamp.(floor.(Int, d_levels .* (paths .- enc.Smin) ./ (enc.Smax - enc.Smin)),
                      0, d_levels - 1)
    xi = Matrix{Int}(undef, N, M * m)
    @inbounds for t in 1:M, k in 1:m
        shift = m - k           # k=1 → MSB
        site  = (t - 1) * m + k
        @simd for i in 1:N
            xi[i, site] = ((buckets[i, t] >> shift) & 1) + 1
        end
    end
    return xi
end

function decode_paths(enc::BinaryEncoder, xi::AbstractMatrix)
    N       = size(xi, 1)
    n_sites = size(xi, 2)
    m       = enc.m
    @assert n_sites % m == 0
    M        = n_sites ÷ m
    d_levels = 2^m
    paths    = Matrix{Float64}(undef, N, M)
    @inbounds for t in 1:M, i in 1:N
        bucket = 0
        for k in 1:m
            shift = m - k
            bit   = xi[i, (t - 1) * m + k] - 1
            bucket |= (bit << shift)
        end
        paths[i, t] = enc.Smin + (bucket / (d_levels - 1)) * (enc.Smax - enc.Smin)
    end
    return paths
end

# ─── TrigEncoder ──────────────────────────────────────────────────────────────

"""
    TrigEncoder(m, d_features = 4; Smin = NaN, Smax = NaN, angle_mode = :discrete_cell_pi)

Bucket prices into `K = 2^m` levels (same as BasisEncoder), but the MPS uses
`d_features` Fourier features (`cos(hθ)`, `sin(hθ)` for `h = 1:d_features÷2`).

Returns a `K × d_features` feature matrix from `feature_map(enc)` that activates
Gram-weighted training (`G = Φ'Φ`). Lower bond dimension at the same capacity.

`angle_mode`:
* `:discrete_cell_pi` (default) — `θ_k = π(k−½)/K` on `(0,π)`.
* `:dense_torus`                — `θ_k = 2π(k−½)/K` on `(0,2π)`.
"""
mutable struct TrigEncoder <: PathEncoder
    m::Int
    d_features::Int
    Smin::Float64
    Smax::Float64
    angle_mode::Symbol
end

function TrigEncoder(
    m::Int, d_features::Int = 4;
    Smin::Real = NaN, Smax::Real = NaN,
    angle_mode::Symbol = :discrete_cell_pi,
)
    @assert d_features >= 2 && iseven(d_features) "d_features must be an even integer ≥ 2"
    @assert angle_mode in (:discrete_cell_pi, :dense_torus) "angle_mode must be :discrete_cell_pi or :dense_torus"
    TrigEncoder(m, d_features, Float64(Smin), Float64(Smax), angle_mode)
end

chain_length(::TrigEncoder, M::Int) = M
site_dim(enc::TrigEncoder)          = enc.d_features

function fit_grid!(enc::TrigEncoder, paths::AbstractMatrix)
    enc.Smin = Float64(minimum(paths))
    enc.Smax = Float64(maximum(paths))
    return enc
end

function encode_paths(enc::TrigEncoder, paths::AbstractMatrix)
    @assert isfinite(enc.Smin) && isfinite(enc.Smax) "Call fit_grid!(enc, paths) first."
    d  = 2^enc.m
    xi = clamp.(floor.(Int, (d - 1) .* (paths .- enc.Smin) ./ (enc.Smax - enc.Smin)) .+ 1, 1, d)
    return xi
end

function decode_paths(enc::TrigEncoder, xi::AbstractMatrix)
    d = 2^enc.m
    return enc.Smin .+ ((xi .- 1) ./ (d - 1)) .* (enc.Smax - enc.Smin)
end

"""Build `K × d_features` Fourier-feature matrix."""
function _trig_phi_matrix(m_bits::Int, d_features::Int, angle_mode::Symbol = :discrete_cell_pi)
    K      = 2^m_bits
    n_harm = d_features ÷ 2
    Φ      = zeros(Float64, K, d_features)
    @inbounds for k in 1:K
        θ = if angle_mode === :dense_torus
            2 * π * (k - 0.5) / K
        else
            π * (k - 0.5) / K
        end
        c = 1
        for h in 1:n_harm
            Φ[k, c]     = cos(h * θ); c += 1
            Φ[k, c]     = sin(h * θ); c += 1
        end
    end
    return Φ
end

function feature_map(enc::TrigEncoder)
    return _trig_phi_matrix(enc.m, enc.d_features, enc.angle_mode)
end

# ─── Sampling ─────────────────────────────────────────────────────────────────

"""
    sample_paths(enc, mps, n_samples; seed = 0) -> (paths, xi)

Sequential conditional sampling from `p(xi) = |Ψ(xi)|²/Z`, decoded to
real-valued paths. Returns `(paths::Matrix{Float64}, xi::Matrix{Int})`.

Delegates to `sample_paths_feature_map` when `feature_map(enc) ≠ nothing`.
For `Float32` MPS, right environments are computed in `Float64` to prevent
overflow in large-bond conditional probabilities.
"""
function sample_paths(
    enc::PathEncoder,
    mps::Vector{<:AbstractArray{T,3}},
    n_samples::Int;
    seed::Int = 0,
) where {T<:Real}
    phi = feature_map(enc)
    if phi !== nothing
        Phi    = Matrix{T}(phi)
        xi_out = sample_paths_feature_map(mps, Phi, n_samples; seed = seed)
        return decode_paths(enc, xi_out), xi_out
    end
    Ml = length(mps)
    d  = size(mps[1], 2)
    @assert site_dim(enc) == d

    if T === Float32
        Tf   = Float64
        mF   = [Tf.(A) for A in mps]
        Renv = Vector{Matrix{Tf}}(undef, Ml + 2)
        Renv[Ml + 2] = Matrix{Tf}(I, 1, 1)
        for j in Ml:-1:1
            A  = mF[j]
            Dl = size(A, 1)
            E  = zeros(Tf, Dl, Dl)
            for σ in 1:d
                Mσ = A[:, σ, :]
                E += Mσ * Renv[j + 2] * Mσ'
            end
            Renv[j + 1] = E
        end
        xi_out = Matrix{Int}(undef, n_samples, Ml)
        Threads.@threads for i in 1:n_samples
            rng = MersenneTwister(seed + i)
            ell = Matrix{Tf}(I, 1, 1)
            for j in 1:Ml
                A  = mF[j]
                pv = Vector{Float64}(undef, d)
                for σ in 1:d
                    v     = ell * A[:, σ, :]
                    pv[σ] = max((v * Renv[j + 2] * v')[1], 0.0)
                end
                s = sum(pv)
                if !(s > 0) || !isfinite(s)
                    pv .= 1.0 / d
                else
                    pv ./= s; pv ./= sum(pv)
                end
                σ_pick = rand(rng, Categorical(pv))
                xi_out[i, j] = σ_pick
                ell = ell * A[:, σ_pick, :]
            end
        end
        return decode_paths(enc, xi_out), xi_out
    end

    Renv = Vector{Matrix{T}}(undef, Ml + 2)
    Renv[Ml + 2] = Matrix{T}(I, 1, 1)
    for j in Ml:-1:1
        A  = mps[j]
        Dl = size(A, 1)
        E  = zeros(T, Dl, Dl)
        for σ in 1:d
            Mσ = A[:, σ, :]
            E += Mσ * Renv[j + 2] * Mσ'
        end
        Renv[j + 1] = E
    end
    xi_out = Matrix{Int}(undef, n_samples, Ml)
    Threads.@threads for i in 1:n_samples
        rng = MersenneTwister(seed + i)
        ell = Matrix{T}(I, 1, 1)
        for j in 1:Ml
            A  = mps[j]
            pv = Vector{Float64}(undef, d)
            for σ in 1:d
                v     = ell * A[:, σ, :]
                pv[σ] = max(Float64((v * Renv[j + 2] * v')[1]), 0.0)
            end
            s = sum(pv)
            if !(s > 0) || !isfinite(s)
                pv .= 1.0 / d
            else
                pv ./= s; pv ./= sum(pv)
            end
            σ_pick = rand(rng, Categorical(pv))
            xi_out[i, j] = σ_pick
            ell = ell * A[:, σ_pick, :]
        end
    end
    return decode_paths(enc, xi_out), xi_out
end

# ─── Classification helpers ───────────────────────────────────────────────────

"""MPS chain length for joint `p(xi, y)`: path sites + 1 label site."""
classification_chain_length(enc::PathEncoder, M::Int) = chain_length(enc, M) + 1

"""Physical leg dimension for the label site (one-hot class index)."""
label_site_dim(n_classes::Int) = n_classes

"""
    encode_labeled_paths(enc, paths, labels; n_classes = 2) -> xi

Encode `paths` with `enc`, then append the integer label `y ∈ 1:n_classes`
as an extra column. Returns `xi` of size `(N, classification_chain_length)`.
"""
function encode_labeled_paths(
    enc::PathEncoder,
    paths::AbstractMatrix,
    labels::AbstractVector{<:Integer};
    n_classes::Int = 2,
)
    xi_path = encode_paths(enc, paths)
    N, Ml_path = size(xi_path)
    @assert length(labels) == N
    xi = Matrix{Int}(undef, N, Ml_path + 1)
    xi[:, 1:Ml_path] = xi_path
    @inbounds for i in 1:N
        y = Int(labels[i])
        @assert 1 <= y <= n_classes "label $y outside 1:n_classes=$n_classes"
        xi[i, Ml_path + 1] = y
    end
    return xi
end

# ─── Utility ──────────────────────────────────────────────────────────────────

"""
    encoder_summary(enc, M, D_max) -> NamedTuple

Pretty-print the encoder's chain length, site dimension and approximate
parameter budget for `M` Heston timesteps and bond cap `D_max`.
"""
function encoder_summary(enc::PathEncoder, M::Int, D_max::Int)
    Ml       = chain_length(enc, M)
    d        = site_dim(enc)
    n_int    = max(Ml - 2, 0)
    params   = 2 * d * D_max + n_int * d * D_max^2
    info     = (encoder      = string(typeof(enc)),
                M            = M,
                chain_length = Ml,
                site_dim     = d,
                D_max        = D_max,
                params       = params)
    @info "Encoder" encoder=info.encoder M=M chain_length=Ml site_dim=d D_max=D_max params=params
    return info
end

end # module Encoders
