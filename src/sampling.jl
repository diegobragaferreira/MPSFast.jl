# sampling.jl — Sequential conditional sampling from a Born-machine MPS.

"""Draw one sample from an unnormalised nonneg weight vector `pv` (length `K`)."""
function _sample_from_pv!(rng::AbstractRNG, pv::Vector{Float64}, K::Int)
    s = sum(pv)
    if !(s > 0) || !isfinite(s)
        return rand(rng, 1:K)
    end
    r = rand(rng) * s
    c = 0.0
    @inbounds for k in 1:K
        c += pv[k]
        r <= c && return k
    end
    return K
end

"""Split `1:n` into at most `nt` contiguous ranges (equal lengths ±1)."""
function _partition_1_to_n(n::Int, nt::Int)
    n <= 0 && return UnitRange{Int}[]
    nt = max(1, min(nt, n))
    ranges = Vector{UnitRange{Int}}(undef, nt)
    base, rem = divrem(n, nt)
    start = 1
    @inbounds for t in 1:nt
        len   = base + (t <= rem ? 1 : 0)
        stop  = start + len - 1
        ranges[t] = start:stop
        start = stop + 1
    end
    return ranges
end

"""
    sample_paths_feature_map(mps, Phi, n_samples; seed = 0) -> xi::Matrix{Int}

Sequential conditional sampling when each site uses the low-rank lift defined by
rows `Phi[k,:]` (`K` buckets, `d` features). The output columns are bucket
indices in `1:K`, suitable for `Encoders.decode_paths` on a `TrigEncoder`.

For `Float32` MPS the Gram right environments and the left prefix `ell` are
accumulated in `Float64` (Float32 overflows for long chains / large bonds).

Parallelism: samples are split into contiguous chunks; each chunk owns its
own scratch matrix (no TLS sharing, which is unsafe under task migration).
"""
function sample_paths_feature_map(
    mps::Vector{Array{T,3}}, Phi::AbstractMatrix{<:Real}, n_samples::Int;
    seed::Int = 0,
) where {T<:Real}
    PhiT   = Phi isa Matrix{T} ? Phi : Matrix{T}(Phi)
    Ml     = length(mps)
    K, d_feat = size(PhiT)
    @assert size(mps[1], 2) == d_feat
    Dmax = maximum(max(size(A, 1), size(A, 3)) for A in mps)

    if T === Float32
        Tf  = Float64
        mF  = [Tf.(A) for A in mps]
        PhiF = Matrix{Tf}(PhiT)
        GF   = PhiF' * PhiF
        tM   = Matrix{Tf}(undef, Dmax, Dmax)
        tA   = Matrix{Tf}(undef, Dmax, Dmax)
        Renv = Vector{Matrix{Tf}}(undef, Ml + 2)
        Renv[Ml + 2] = Matrix{Tf}(I, 1, 1)
        @inbounds for j in Ml:-1:1
            Dl = size(mF[j], 1)
            Dr = size(mF[j], 3)
            Renv[j + 1] = Matrix{Tf}(undef, Dl, Dl)
            _transfer_right_G!(Renv[j + 1], Renv[j + 2], mF[j], GF,
                               view(tM, 1:Dl, 1:Dr), view(tA, 1:Dl, 1:Dl))
        end
        xi_out = Matrix{Int}(undef, n_samples, Ml)
        ranges = _partition_1_to_n(n_samples, Threads.nthreads())
        Threads.@threads for ti in 1:length(ranges)
            r  = ranges[ti]
            mk = Matrix{Tf}(undef, Dmax, Dmax)
            @inbounds for i in r
                rng = MersenneTwister(seed + i)
                ell = Matrix{Tf}(I, 1, 1)
                @inbounds for j in 1:Ml
                    A    = mF[j]
                    Dl, dloc, Dr = size(A)
                    pv   = Vector{Float64}(undef, K)
                    @inbounds for k in 1:K
                        fill!(view(mk, 1:Dl, 1:Dr), zero(Tf))
                        @inbounds for σ in 1:dloc
                            ph = PhiF[k, σ]
                            iszero(ph) && continue
                            axpy!(ph, view(A, :, σ, :), view(mk, 1:Dl, 1:Dr))
                        end
                        v      = ell * view(mk, 1:Dl, 1:Dr)
                        pv[k]  = max(Float64((v * Renv[j + 2] * v')[1, 1]), 0.0)
                    end
                    k_pick = _sample_from_pv!(rng, pv, K)
                    xi_out[i, j] = k_pick
                    fill!(view(mk, 1:Dl, 1:Dr), zero(Tf))
                    @inbounds for σ in 1:dloc
                        ph = PhiF[k_pick, σ]
                        iszero(ph) && continue
                        axpy!(ph, view(A, :, σ, :), view(mk, 1:Dl, 1:Dr))
                    end
                    ell = ell * view(mk, 1:Dl, 1:Dr)
                end
            end
        end
        return xi_out
    end

    G      = PhiT' * PhiT
    tmp_dd = Matrix{T}(undef, Dmax, Dmax)
    mid_dr = Matrix{T}(undef, Dmax, Dmax)
    acc_dl = Matrix{T}(undef, Dmax, Dmax)
    Renv   = Vector{Matrix{T}}(undef, Ml + 2)
    Renv[Ml + 2] = Matrix{T}(I, 1, 1)
    @inbounds for j in Ml:-1:1
        Dl = size(mps[j], 1)
        Dr = size(mps[j], 3)
        Renv[j + 1] = Matrix{T}(undef, Dl, Dl)
        _transfer_right_G!(Renv[j + 1], Renv[j + 2], mps[j], G,
                           view(mid_dr, 1:Dl, 1:Dr), view(acc_dl, 1:Dl, 1:Dl))
    end
    xi_out = Matrix{Int}(undef, n_samples, Ml)
    ranges = _partition_1_to_n(n_samples, Threads.nthreads())
    Threads.@threads for ti in 1:length(ranges)
        r  = ranges[ti]
        mk = Matrix{T}(undef, Dmax, Dmax)
        @inbounds for i in r
            rng = MersenneTwister(seed + i)
            ell = Matrix{T}(I, 1, 1)
            @inbounds for j in 1:Ml
                A    = mps[j]
                Dl, dloc, Dr = size(A)
                pv   = Vector{Float64}(undef, K)
                @inbounds for k in 1:K
                    fill!(view(mk, 1:Dl, 1:Dr), zero(T))
                    @inbounds for σ in 1:dloc
                        ph = PhiT[k, σ]
                        iszero(ph) && continue
                        axpy!(ph, view(A, :, σ, :), view(mk, 1:Dl, 1:Dr))
                    end
                    v     = ell * view(mk, 1:Dl, 1:Dr)
                    pv[k] = max(Float64((v * Renv[j + 2] * v')[1, 1]), 0.0)
                end
                k_pick = _sample_from_pv!(rng, pv, K)
                xi_out[i, j] = k_pick
                fill!(view(mk, 1:Dl, 1:Dr), zero(T))
                @inbounds for σ in 1:dloc
                    ph = PhiT[k_pick, σ]
                    iszero(ph) && continue
                    axpy!(ph, view(A, :, σ, :), view(mk, 1:Dl, 1:Dr))
                end
                ell = ell * view(mk, 1:Dl, 1:Dr)
            end
        end
    end
    return xi_out
end
