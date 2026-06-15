# training.jl — DMRG-style NLL training: workspace, boundary vectors,
#               NLL gradient, truncated SVD, Adam, bond-pair updates, full sweep.
#
# Key design choices:
#   - σ-binned gather → BLAS GEMM → scatter avoids non-strided SubArray fallback
#     that routes mul! through slow generic loops.
#   - Fresh Array{T,4} allocations per nll_gradient! call ensure reshape(arr,…)
#     yields a true contiguous Matrix for BLAS (not a ReshapedArray).
#   - Float32 MPS + Gram: norm environments are stored in Float64 (env_L64/R64)
#     to prevent overflow in Z = ‖Ψ‖².

# ─── Workspace ────────────────────────────────────────────────────────────────

"""
    TrainWorkspace{T}

Persistent scratch buffers for one full training session.

Allocated once by `train_mps!` or explicitly via `TrainWorkspace(T, Nd, d, Dmax;
feature_phi)`. Passed into `nll_gradient!` and `update_pair!` to avoid repeated
allocations in the hot path.
"""
mutable struct TrainWorkspace{T<:Real}
    Nd::Int
    d::Int
    Dmax::Int
    Lv_scaled::Matrix{T}
    psi::Vector{T}
    bin_idx::Vector{Vector{Int}}
    bins_per_site::Vector{Vector{Vector{Int}}}
    active_site_buckets::Vector{Vector{Int}}
    bins_xi_id::UInt64
    gather_in::Matrix{T}
    gather_out::Matrix{T}
    Lv_ext::Matrix{T}
    Rv_ext::Matrix{T}
    feature_phi::Union{Nothing, Matrix{T}}
    gram::Union{Nothing, Matrix{T}}
    gtmp_dd::Matrix{T}
    gmid_dr::Matrix{T}
    gacc_dl::Matrix{T}
    feat_mk::Matrix{T}
    pair_key_buf::Vector{Int}
    env_L64::Vector{Matrix{Float64}}
    env_R64::Vector{Matrix{Float64}}
end

function TrainWorkspace(
    ::Type{T}, Nd::Int, d::Int, Dmax::Int;
    feature_phi::Union{Nothing, AbstractMatrix} = nothing,
) where {T<:Real}
    bin0     = max(d * d, 64)
    bins     = [Int[] for _ in 1:bin0]
    for v in bins; sizehint!(v, max(8, Nd ÷ max(d * d, 1) + 4)); end
    phi_mat  = feature_phi === nothing ? nothing : Matrix{T}(feature_phi)
    gram_mat = phi_mat === nothing ? nothing : Matrix{T}(phi_mat' * phi_mat)
    TrainWorkspace{T}(
        Nd, d, Dmax,
        Matrix{T}(undef, Nd, Dmax),
        Vector{T}(undef, Nd),
        bins,
        Vector{Vector{Vector{Int}}}(),
        Vector{Vector{Int}}(),
        UInt64(0),
        Matrix{T}(undef, Dmax, Nd),
        Matrix{T}(undef, Dmax, Nd),
        Matrix{T}(undef, Nd, Dmax),
        Matrix{T}(undef, Nd, Dmax),
        phi_mat,
        gram_mat,
        Matrix{T}(undef, Dmax, Dmax),
        Matrix{T}(undef, Dmax, Dmax),
        Matrix{T}(undef, Dmax, Dmax),
        Matrix{T}(undef, Dmax, Dmax),
        Vector{Int}(undef, Nd),
        Matrix{Float64}[],
        Matrix{Float64}[],
    )
end

function _fill_gram_norm_env_float64!(
    mps::Vector{Array{Float32,3}}, ws::TrainWorkspace{Float32},
)
    Lf, Rf = _norm_environments_gram_f32_sweep(mps, ws.gram::AbstractMatrix{Float32}, ws.Dmax)
    ws.env_L64 = Lf
    ws.env_R64 = Rf
    return nothing
end

function _ensure_bin_idx_capacity!(ws::TrainWorkspace, need::Int)
    while length(ws.bin_idx) < need
        push!(ws.bin_idx, Int[])
    end
    return
end

"""Cache `bins_per_site[k][bucket]` = sorted sample ids with `xi_data[i,k] == bucket`.
Recomputed only when `xi_data`'s `objectid` changes."""
function _ensure_bins!(ws::TrainWorkspace{T}, xi_data) where {T<:Real}
    Nd, M = size(xi_data)
    @assert Nd == ws.Nd "TrainWorkspace Nd=$(ws.Nd) but xi_data has Nd=$Nd"
    id = objectid(xi_data)
    if ws.bins_xi_id == id && length(ws.bins_per_site) == M && length(ws.active_site_buckets) == M
        return
    end
    max_bucket = max(Int(maximum(xi_data)), ws.d)
    bps = Vector{Vector{Vector{Int}}}(undef, M)
    occ = Vector{Vector{Int}}(undef, M)
    @inbounds for k in 1:M
        counts = zeros(Int, max_bucket)
        for i in 1:Nd
            counts[xi_data[i, k]] += 1
        end
        act = Int[]
        sizehint!(act, min(Nd, max_bucket))
        @inbounds for σ in 1:max_bucket
            counts[σ] > 0 && push!(act, σ)
        end
        occ[k] = act
        bins = Vector{Vector{Int}}(undef, max_bucket)
        for σ in 1:max_bucket
            bins[σ] = Vector{Int}(undef, counts[σ])
        end
        ptrs = ones(Int, max_bucket)
        for i in 1:Nd
            σ = xi_data[i, k]
            bins[σ][ptrs[σ]] = i
            ptrs[σ] += 1
        end
        bps[k] = bins
        @assert !isempty(occ[k]) "Site $k: no occupied buckets (check xi_data)."
    end
    ws.bins_per_site       = bps
    ws.active_site_buckets = occ
    ws.bins_xi_id          = id
    return
end

# ─── Boundary-vector extension (σ-binned gather → BLAS → scatter) ────────────

@inline function _extend_lv_ws!(
    Lv_out::AbstractMatrix{T}, Lv_in::AbstractMatrix{T},
    A::AbstractArray{T,3}, bins::Vector{Vector{Int}},
    ws::TrainWorkspace{T},
) where {T<:Real}
    Nd, Dl = size(Lv_in)
    Dl2, d, Dr = size(A)
    @assert Dl == Dl2 && size(Lv_out) == (Nd, Dr)
    @inbounds for σ in 1:d
        ids = bins[σ]
        n_b = length(ids)
        n_b == 0 && continue
        Gin  = view(ws.gather_in,  1:Dl, 1:n_b)
        Gout = view(ws.gather_out, 1:Dr, 1:n_b)
        @inbounds for k in 1:n_b
            i = ids[k]
            @simd for α in 1:Dl
                Gin[α, k] = Lv_in[i, α]
            end
        end
        mul!(Gout, transpose(view(A, :, σ, :)), Gin)
        @inbounds for k in 1:n_b
            i = ids[k]
            @simd for β in 1:Dr
                Lv_out[i, β] = Gout[β, k]
            end
        end
    end
    return Lv_out
end

@inline function _extend_rv_ws!(
    Rv_out::AbstractMatrix{T}, Rv_in::AbstractMatrix{T},
    A::AbstractArray{T,3}, bins::Vector{Vector{Int}},
    ws::TrainWorkspace{T},
) where {T<:Real}
    Nd, Dr = size(Rv_in)
    Dl, d, Dr2 = size(A)
    @assert Dr == Dr2 && size(Rv_out) == (Nd, Dl)
    @inbounds for σ in 1:d
        ids = bins[σ]
        n_b = length(ids)
        n_b == 0 && continue
        Gin  = view(ws.gather_in,  1:Dr, 1:n_b)
        Gout = view(ws.gather_out, 1:Dl, 1:n_b)
        @inbounds for k in 1:n_b
            i = ids[k]
            @simd for β in 1:Dr
                Gin[β, k] = Rv_in[i, β]
            end
        end
        mul!(Gout, view(A, :, σ, :), Gin)
        @inbounds for k in 1:n_b
            i = ids[k]
            @simd for α in 1:Dl
                Rv_out[i, α] = Gout[α, k]
            end
        end
    end
    return Rv_out
end

function _extend_lv_phi!(
    Lv_out::AbstractMatrix{T}, Lv_in::AbstractMatrix{T},
    A::AbstractArray{T,3}, bins_K::Vector{Vector{Int}},
    Phi::AbstractMatrix{T}, ws::TrainWorkspace{T}, site::Int,
) where {T<:Real}
    Nd, Dl = size(Lv_in)
    Dl2, d, Dr = size(A)
    @assert Dl == Dl2 && size(Lv_out) == (Nd, Dr)
    mk = ws.feat_mk
    @inbounds for k in ws.active_site_buckets[site]
        ids = bins_K[k]
        n_b = length(ids)
        n_b == 0 && continue
        fill!(view(mk, 1:Dl, 1:Dr), zero(T))
        @inbounds for σ in 1:d
            ph = Phi[k, σ]
            iszero(ph) && continue
            axpy!(ph, view(A, :, σ, :), view(mk, 1:Dl, 1:Dr))
        end
        Gin  = view(ws.gather_in,  1:Dl, 1:n_b)
        Gout = view(ws.gather_out, 1:Dr, 1:n_b)
        @inbounds for kk in 1:n_b
            i = ids[kk]
            @simd for α in 1:Dl
                Gin[α, kk] = Lv_in[i, α]
            end
        end
        mul!(Gout, transpose(view(mk, 1:Dl, 1:Dr)), Gin)
        @inbounds for kk in 1:n_b
            i = ids[kk]
            @simd for β in 1:Dr
                Lv_out[i, β] = Gout[β, kk]
            end
        end
    end
    return Lv_out
end

function _extend_rv_phi!(
    Rv_out::AbstractMatrix{T}, Rv_in::AbstractMatrix{T},
    A::AbstractArray{T,3}, bins_K::Vector{Vector{Int}},
    Phi::AbstractMatrix{T}, ws::TrainWorkspace{T}, site::Int,
) where {T<:Real}
    Nd, Dr = size(Rv_in)
    Dl, d, Dr2 = size(A)
    @assert Dr == Dr2 && size(Rv_out) == (Nd, Dl)
    mk = ws.feat_mk
    @inbounds for k in ws.active_site_buckets[site]
        ids = bins_K[k]
        n_b = length(ids)
        n_b == 0 && continue
        fill!(view(mk, 1:Dl, 1:Dr), zero(T))
        @inbounds for σ in 1:d
            ph = Phi[k, σ]
            iszero(ph) && continue
            axpy!(ph, view(A, :, σ, :), view(mk, 1:Dl, 1:Dr))
        end
        Gin  = view(ws.gather_in,  1:Dr, 1:n_b)
        Gout = view(ws.gather_out, 1:Dl, 1:n_b)
        @inbounds for kk in 1:n_b
            i = ids[kk]
            @simd for β in 1:Dr
                Gin[β, kk] = Rv_in[i, β]
            end
        end
        mul!(Gout, view(mk, 1:Dl, 1:Dr), Gin)
        @inbounds for kk in 1:n_b
            i = ids[kk]
            @simd for α in 1:Dl
                Rv_out[i, α] = Gout[α, kk]
            end
        end
    end
    return Rv_out
end

# ─── Boundary vectors ─────────────────────────────────────────────────────────

"""Allocating wrapper used only outside the hot path (smoke tests, debugging)."""
function boundary_vectors(mps::Vector{Array{T,3}}, xi_data, j) where {T<:Real}
    Nd   = size(xi_data, 1)
    d    = size(mps[1], 2)
    Dmax = _max_mps_bond(mps)
    ws   = TrainWorkspace(T, Nd, d, Dmax)
    return boundary_vectors(ws, mps, xi_data, j)
end

function boundary_vectors(
    ws::TrainWorkspace{T}, mps::Vector{Array{T,3}}, xi_data, j,
) where {T<:Real}
    _ensure_bins!(ws, xi_data)
    Nd = size(xi_data, 1)
    Ml = length(mps)
    Phi = ws.feature_phi
    Lv = ones(T, Nd, 1)
    if Phi === nothing
        @inbounds for k in 1:(j-1)
            Dr  = size(mps[k], 3)
            Lv2 = Matrix{T}(undef, Nd, Dr)
            _extend_lv_ws!(Lv2, Lv, mps[k], ws.bins_per_site[k], ws)
            Lv = Lv2
        end
    else
        @inbounds for k in 1:(j-1)
            Dr  = size(mps[k], 3)
            Lv2 = Matrix{T}(undef, Nd, Dr)
            _extend_lv_phi!(Lv2, Lv, mps[k], ws.bins_per_site[k], Phi, ws, k)
            Lv = Lv2
        end
    end
    Rv = ones(T, Nd, 1)
    if Phi === nothing
        @inbounds for k in Ml:-1:(j+2)
            Dl  = size(mps[k], 1)
            Rv2 = Matrix{T}(undef, Nd, Dl)
            _extend_rv_ws!(Rv2, Rv, mps[k], ws.bins_per_site[k], ws)
            Rv = Rv2
        end
    else
        @inbounds for k in Ml:-1:(j+2)
            Dl  = size(mps[k], 1)
            Rv2 = Matrix{T}(undef, Nd, Dl)
            _extend_rv_phi!(Rv2, Rv, mps[k], ws.bins_per_site[k], Phi, ws, k)
            Rv = Rv2
        end
    end
    psi = copy(_compute_psi!(ws, mps, xi_data, j, Lv, Rv))
    return Lv, Rv, psi
end

function _compute_psi!(
    ws::TrainWorkspace{T}, mps, xi_data, j,
    Lv::AbstractMatrix{T}, Rv::AbstractMatrix{T},
) where {T<:Real}
    _ensure_bins!(ws, xi_data)
    Aj   = mps[j];   Ajp1 = mps[j+1]
    _,   _,  Dm  = size(Aj)
    Nd   = size(xi_data, 1)
    Lv_ext = view(ws.Lv_ext, 1:Nd, 1:Dm)
    Rv_ext = view(ws.Rv_ext, 1:Nd, 1:Dm)
    Phi    = ws.feature_phi
    if Phi === nothing
        _extend_lv_ws!(Lv_ext, Lv, Aj,   ws.bins_per_site[j],   ws)
        _extend_rv_ws!(Rv_ext, Rv, Ajp1, ws.bins_per_site[j+1], ws)
    else
        _extend_lv_phi!(Lv_ext, Lv, Aj,   ws.bins_per_site[j],   Phi, ws, j)
        _extend_rv_phi!(Rv_ext, Rv, Ajp1, ws.bins_per_site[j+1], Phi, ws, j + 1)
    end
    psi = ws.psi
    @inbounds for i in 1:Nd
        s = zero(T)
        @simd for γ in 1:Dm
            s += Lv_ext[i, γ] * Rv_ext[i, γ]
        end
        psi[i] = s
    end
    return psi
end

function _compute_psi(
    mps, xi_data, j, Lv::AbstractMatrix{T}, Rv::AbstractMatrix{T},
) where {T<:Real}
    Nd   = size(xi_data, 1)
    d    = size(mps[1], 2)
    Dmax = _max_mps_bond(mps)
    ws   = TrainWorkspace(T, Nd, d, Dmax)
    return copy(_compute_psi!(ws, mps, xi_data, j, Lv, Rv))
end

function boundary_vectors_right_psi!(
    ws::TrainWorkspace{T}, mps::Vector{Array{T,3}}, xi_data, j, Lv::Matrix{T},
) where {T<:Real}
    _ensure_bins!(ws, xi_data)
    Nd  = size(xi_data, 1)
    Ml  = length(mps)
    Phi = ws.feature_phi
    Rv  = ones(T, Nd, 1)
    if Phi === nothing
        @inbounds for k in Ml:-1:(j+2)
            Dl  = size(mps[k], 1)
            Rv2 = Matrix{T}(undef, Nd, Dl)
            _extend_rv_ws!(Rv2, Rv, mps[k], ws.bins_per_site[k], ws)
            Rv = Rv2
        end
    else
        @inbounds for k in Ml:-1:(j+2)
            Dl  = size(mps[k], 1)
            Rv2 = Matrix{T}(undef, Nd, Dl)
            _extend_rv_phi!(Rv2, Rv, mps[k], ws.bins_per_site[k], Phi, ws, k)
            Rv = Rv2
        end
    end
    psi = _compute_psi!(ws, mps, xi_data, j, Lv, Rv)
    return Lv, Rv, psi
end

"""Contract sites 1…(j−1) into left boundary vector (j==1 → trivial 1×1)."""
function lv_prefix(
    ws::TrainWorkspace{T}, mps::Vector{Array{T,3}}, xi_data, j::Int,
) where {T<:Real}
    Nd = size(xi_data, 1)
    j <= 1 && return ones(T, Nd, 1)
    _ensure_bins!(ws, xi_data)
    Lv = ones(T, Nd, 1)
    @inbounds for jj in 1:(j-1)
        Lv = extend_lv_after_bond!(ws, mps, xi_data, jj, Lv)
    end
    return Lv
end

function extend_lv_after_bond!(
    ws::TrainWorkspace{T}, mps::Vector{Array{T,3}}, xi_data, j::Int, Lv::Matrix{T},
) where {T<:Real}
    _ensure_bins!(ws, xi_data)
    Nd     = size(Lv, 1)
    Dr     = size(mps[j], 3)
    Lv_out = Matrix{T}(undef, Nd, Dr)
    Phi    = ws.feature_phi
    if Phi === nothing
        _extend_lv_ws!(Lv_out, Lv, mps[j], ws.bins_per_site[j], ws)
    else
        _extend_lv_phi!(Lv_out, Lv, mps[j], ws.bins_per_site[j], Phi, ws, j)
    end
    return Lv_out
end

# ─── NLL gradient ─────────────────────────────────────────────────────────────

"""
    nll_gradient!(ws, mps, xi_data, j, Lenv, Renv; Lv_carry) -> (grad, B, Z_val, Lv, Rv, psi)

Gradient of the negative log-likelihood with respect to the merged two-site
tensor `B = A_j ⊗ A_{j+1}` at bond `j`.

All heavy contractions reduce to three BLAS GEMMs (L·B, result·R, and the
σ-binned gather/scatter passes). Uses cached site-bins from `ws`.
"""
function nll_gradient!(
    ws::TrainWorkspace{T}, mps::Vector{Array{T,3}}, xi_data, j, Lenv, Renv;
    Lv_carry = nothing,
) where {T<:Real}
    _ensure_bins!(ws, xi_data)
    Aj   = mps[j];   Ajp1 = mps[j+1]
    Dl,  d,  Dm  = size(Aj)
    Dm2, d2, Dr  = size(Ajp1)
    @assert Dm == Dm2   # bond dim must match; physical dims may differ (e.g. label site)
    Nd = size(xi_data, 1)

    B   = Array{T, 4}(undef, Dl, d, d2, Dr)
    Y   = Array{T, 4}(undef, Dl, d, d2, Dr)
    LBR = Array{T, 4}(undef, Dl, d, d2, Dr)
    Bmat         = reshape(B,   Dl * d,      d2 * Dr)
    Bmat_LB_in   = reshape(B,   Dl,          d * d2 * Dr)
    Bmat_LB_out  = reshape(Y,   Dl,          d * d2 * Dr)
    Ymat_BR_in   = reshape(Y,   Dl * d * d2, Dr)
    Ymat_BR_out  = reshape(LBR, Dl * d * d2, Dr)
    Aj_mat       = reshape(Aj,   Dl * d, Dm)
    Ajp1_mat     = reshape(Ajp1, Dm,     d2 * Dr)
    mul!(Bmat, Aj_mat, Ajp1_mat)

    Ln = Lenv[j]
    Rn = Renv[j+3]
    gram_z64   = Ref(false)
    Z64_acc    = Ref(0.0)
    LBRf_acc   = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
    if ws.gram !== nothing && T === Float32
        Bf  = Matrix{Float64}(undef, Dl, d * d2 * Dr)
        copyto!(Bf, Bmat_LB_in)
        Yf  = Matrix{Float64}(undef, Dl, d * d2 * Dr)
        mul!(Yf, Ln, Bf)
        LBRf = Matrix{Float64}(undef, Dl * d * d2, Dr)
        mul!(LBRf, reshape(Yf, Dl * d * d2, Dr), Rn)
        Z64_acc[]  = dot(Base.vec(Bf), Base.vec(LBRf))
        gram_z64[] = true
        LBRf_acc[] = LBRf
    else
        mul!(Bmat_LB_out, Ln, Bmat_LB_in)
        mul!(Ymat_BR_out, Ymat_BR_in, Rn)
    end

    if Lv_carry === nothing
        Lv, Rv, psi = boundary_vectors(ws, mps, xi_data, j)
    else
        Lv, Rv, psi = boundary_vectors_right_psi!(ws, mps, xi_data, j, Lv_carry)
    end

    ddata = zeros(T, Dl, d, d2, Dr)
    Phi_fm = ws.feature_phi
    if Phi_fm === nothing
        @inbounds for v in ws.bin_idx; empty!(v); end
        @inbounds for i in 1:Nd
            σi = xi_data[i, j]; τi = xi_data[i, j+1]
            push!(ws.bin_idx[(τi - 1) * d + σi], i)
        end
        @inbounds for k in 1:(d * d2)
            ids = ws.bin_idx[k]
            n_b = length(ids)
            n_b == 0 && continue
            σ = ((k - 1) % d) + 1
            τ = ((k - 1) ÷ d) + 1
            @inbounds for kk in 1:n_b
                i  = ids[kk]
                p  = psi[i]
                iszero(p) && continue
                # Guard: 2/p must not overflow T (subnormals in Float32 cause Inf).
                inv_p_f64 = 2.0 / Float64(p)
                isfinite(inv_p_f64) || continue
                inv_p = T(inv_p_f64)
                isfinite(inv_p) || continue
                @inbounds for β in 1:Dr
                    Rb = inv_p * Rv[i, β]
                    @simd for α in 1:Dl
                        ddata[α, σ, τ, β] += Lv[i, α] * Rb
                    end
                end
            end
        end
    else
        K  = size(Phi_fm, 1)
        pk = ws.pair_key_buf
        @assert length(pk) >= Nd
        @inbounds for i in 1:Nd
            k1 = xi_data[i, j]; k2 = xi_data[i, j+1]
            pk[i] = (k2 - 1) * K + k1
        end
        perm = sortperm(pk)
        Gs   = view(ws.feat_mk, 1:Dl, 1:Dr)
        col  = 1
        @inbounds while col <= Nd
            kk      = pk[perm[col]]
            run_end = col
            @inbounds while run_end < Nd && pk[perm[run_end + 1]] == kk
                run_end += 1
            end
            k1 = ((kk - 1) % K) + 1
            k2 = ((kk - 1) ÷ K) + 1
            fill!(Gs, zero(T))
            @inbounds for t in col:run_end
                i  = perm[t]
                p  = psi[i]
                iszero(p) && continue
                inv_p_f64 = 2.0 / Float64(p)
                isfinite(inv_p_f64) || continue
                inv_p = T(inv_p_f64)
                isfinite(inv_p) || continue
                @inbounds for β in 1:Dr
                    r_scaled = Rv[i, β] * inv_p
                    @simd for α in 1:Dl
                        Gs[α, β] += Lv[i, α] * r_scaled
                    end
                end
            end
            @inbounds for σ in 1:d, τ in 1:d2
                fac = Phi_fm[k1, σ] * Phi_fm[k2, τ]
                iszero(fac) && continue
                @inbounds for β in 1:Dr, α in 1:Dl
                    ddata[α, σ, τ, β] += fac * Gs[α, β]
                end
            end
            col = run_end + 1
        end
    end

    grad = Array{T, 4}(undef, Dl, d, d2, Dr)
    if gram_z64[]
        Z64  = Z64_acc[]
        invZ = inv(max(Z64, 1e-300))
        Z_val = T(Z64)
        LBRf  = LBRf_acc[]::Matrix{Float64}
        Dmat  = reshape(ddata, Dl * d * d2, Dr)
        gr64  = 2.0 .* invZ .* LBRf .- Float64.(Dmat) ./ Float64(Nd)
        copyto!(grad, T.(reshape(gr64, Dl, d, d2, Dr)))
    else
        Z_val = dot(vec(B), vec(LBR))
        invZ  = inv(max(Z_val, T(1e-30)))
        @inbounds @. grad = T(2) * invZ * LBR - ddata / T(Nd)
    end

    return grad, B, Z_val, Lv, Rv, psi
end

# ─── Truncated SVD ────────────────────────────────────────────────────────────

"""Top-`k` SVD. Falls back to full SVD (in Float64 for stability) when the
matrix is small or the iterative solver fails to converge."""
function _truncated_svd(B::AbstractMatrix{T}, k::Int) where {T<:Real}
    Bm  = B isa Matrix ? B : Matrix(B)
    # macOS Accelerate (and some other LAPACK builds) raise "invalid argument #4"
    # when the input contains NaN/Inf.  Replace them with 0 so SVD can still run
    # and the bond is zeroed-out rather than crashing the training loop.
    if !all(isfinite, Bm)
        n_bad = count(!isfinite, Bm)
        @warn "_truncated_svd: $n_bad non-finite entries (NaN/Inf) in matrix of size $(size(Bm)); replacing with 0"
        Bm = map(x -> isfinite(x) ? x : zero(T), Bm)
    end
    m, n = size(Bm)
    if min(m, n) <= max(2 * k, 64)
        F  = svd(Float64.(Bm))
        kk = min(k, length(F.S))
        return T.(F.U[:, 1:kk]), T.(F.S[1:kk]), T.(F.Vt[1:kk, :])
    end
    try
        U, s, V = tsvd(Bm, k)
        return T.(U), T.(s), T.(V')
    catch err
        @warn "TSVD failed (size $(size(Bm)), k=$k): $err — falling back to full SVD"
        F  = svd(Float64.(Bm))
        kk = min(k, length(F.S))
        return T.(F.U[:, 1:kk]), T.(F.S[1:kk]), T.(F.Vt[1:kk, :])
    end
end

# ─── Adam ─────────────────────────────────────────────────────────────────────

mutable struct AdamSlot{T<:Real}
    m::Array{T,4}
    v::Array{T,4}
    t::Int
end

const AdamDict{T} = Dict{NTuple{5,Int}, AdamSlot{T}}

function _adam_step!(
    slot::AdamSlot{T}, grad::AbstractArray{T,4},
    out::AbstractArray{T,4}, η::T,
) where {T<:Real}
    β1 = T(0.9); β2 = T(0.999); εa = T(1e-8)
    slot.t += 1
    @inbounds @. slot.m = β1 * slot.m + (1 - β1) * grad
    @inbounds @. slot.v = β2 * slot.v + (1 - β2) * grad * grad
    bc1 = 1 - β1 ^ slot.t
    bc2 = 1 - β2 ^ slot.t
    @inbounds @. out = η * (slot.m / bc1) / (sqrt(slot.v / bc2) + εa)
    return nothing
end

# ─── Pair update ──────────────────────────────────────────────────────────────

"""
    update_pair!(ws, mps, xi_data, j, η, D_max, ε_cut, Lenv, Renv, adam_state; ...)

One DMRG bond update at bond `(j, j+1)`:
1. Compute NLL gradient of the merged tensor `B = A_j ⊗ A_{j+1}`.
2. Apply in-place Adam step.
3. Truncated SVD back to `D_max` (with absolute floor `ε_cut`).
4. Re-split back into `mps[j]` and `mps[j+1]` (symmetric singular values).
"""
function update_pair!(
    ws::TrainWorkspace{T}, mps::Vector{Array{T,3}}, xi_data, j,
    η, D_max, ε_cut, Lenv, Renv, adam_state::AdamDict{T};
    Lv_carry = nothing, bond_log = nothing,
    epoch::Int = 0, sweep::Symbol = :none,
    d_phys::Int = size(mps[j], 2),
) where {T<:Real}
    Dl = size(mps[j],   1)
    d  = size(mps[j],   2)
    d2 = size(mps[j+1], 2)
    Dr = size(mps[j+1], 3)

    grad, B, _Z, _Lv, _Rv, _ψ =
        nll_gradient!(ws, mps, xi_data, j, Lenv, Renv; Lv_carry = Lv_carry)

    # ── Gradient clipping ─────────────────────────────────────────────────────
    # With Float32 and large D_max the partition function Z can underflow, making
    # invZ huge and blowing up the gradient.  Clip to a global norm of 10 before
    # feeding Adam so that the Adam moment estimates stay finite.
    gnorm = norm(grad)
    if !isfinite(gnorm)
        @warn "update_pair!: NaN/Inf gradient at bond j=$j (Dl=$Dl, d=$d, Dr=$Dr); skipping bond update."
        return nothing
    end
    _GRAD_CLIP = T(10.0)
    if gnorm > _GRAD_CLIP
        @inbounds grad .*= _GRAD_CLIP / gnorm
    end

    key  = (j, Dl, d, d2, Dr)
    slot = get!(adam_state, key) do
        AdamSlot{T}(zeros(T, Dl, d, d2, Dr), zeros(T, Dl, d, d2, Dr), 0)
    end
    step = Array{T, 4}(undef, Dl, d, d2, Dr)
    _adam_step!(slot, grad, step, T(η))

    @inbounds @. B = B - step
    # Guard: if the Adam step produced NaN/Inf (e.g. exploding moments from a
    # previous overflow that slipped through), bail out rather than passing a
    # corrupt matrix to SVD.
    if !all(isfinite, B)
        @warn "update_pair!: NaN/Inf in B after Adam step at bond j=$j; skipping SVD and bond update."
        return nothing
    end
    B_hat = reshape(B, Dl * d, d2 * Dr)

    keep_target = min(D_max, minimum(size(B_hat)))
    if keep_target == 0
        @warn "update_pair!: keep_target=0 at bond j=$j (Dl=$Dl, d=$d, Dr=$Dr, B_hat size=$(size(B_hat)), D_max=$D_max). Skipping bond update."
        return nothing
    end
    U, S, Vt    = _truncated_svd(B_hat, keep_target)
    keep = length(S)
    if keep == 0
        @warn "update_pair!: SVD returned empty S at bond j=$j (keep_target=$keep_target, B_hat size=$(size(B_hat))). Skipping bond update."
        return nothing
    end
    if ε_cut > 0
        keep = max(1, min(sum(S .> T(ε_cut)), keep))
    end
    if sweep === :forward
        # Left-canonical split: site j becomes isometric (U), site j+1 absorbs Σ.
        # Maintains left-canonical form so Lenv[j+1] = I at the next bond.
        mps[j]   = reshape(@view(U[:, 1:keep]),                                      Dl, d,  keep)
        mps[j+1] = reshape(reshape(S[1:keep], keep, 1) .* @view(Vt[1:keep, :]),     keep, d2, Dr)
    elseif sweep === :backward
        # Right-canonical split: site j+1 becomes isometric (Vt), site j absorbs Σ.
        # Maintains right-canonical form so Renv[j+3] = I at the next bond.
        mps[j]   = reshape((@view U[:, 1:keep]) .* reshape(S[1:keep], 1, keep),     Dl, d,  keep)
        mps[j+1] = reshape(@view(Vt[1:keep, :]),                                     keep, d2, Dr)
    else
        # Symmetric split (legacy / non-sweep usage).
        sqS = sqrt.(@view S[1:keep])
        mps[j]   = reshape((@view U[:, 1:keep]) .* reshape(sqS, 1, :), Dl, d,  keep)
        mps[j+1] = reshape(reshape(sqS, :, 1) .* (@view Vt[1:keep, :]), keep, d2, Dr)
    end

    if bond_log !== nothing
        log_bond_spectrum!(bond_log, epoch, sweep, j, S, keep, d_phys)
    end
    return nothing
end

# ─── Norm-env helpers (selects standard vs Gram overload) ─────────────────────

"""
    _canonical_envs_identity(mps)

Return Lenv and Renv both filled with identity matrices of the correct sizes.

Used at the start of canonical sweeps:
- Before forward sweep (after right_canonicalize_mps!):
    Renv[j+3] = I  because all sites to the right are right-isometric →
                   their right transfer maps I → I.
    Lenv[j]   = I  as the starting value; it is grown to I during the sweep
                   by left-canonical transfers.
- Before backward sweep (after left_canonicalize_mps!):
    Lenv[j]   = I  because all sites to the left are left-isometric.
    Renv[j+3] = I  as the starting value; grown by right-canonical transfers.

Computing Lenv/Renv from scratch via _norm_envs_for_train would OVERFLOW because
the left transfer through right-canonical sites can scale by up to Dl per step,
giving Dl^M ≈ 150^30 ≈ 10^63 — way above Float32 max.
"""
function _canonical_envs_identity(mps::Vector{Array{T,3}}) where {T<:Real}
    M = length(mps)
    Lenv = Vector{Matrix{T}}(undef, M + 1)
    Lenv[1] = Matrix{T}(I, 1, 1)
    @inbounds for j in 1:M
        Dr = size(mps[j], 3)
        Lenv[j+1] = Matrix{T}(I, Dr, Dr)
    end
    Renv = Vector{Matrix{T}}(undef, M + 2)
    Renv[M+2] = Matrix{T}(I, 1, 1)
    @inbounds for j in M:-1:1
        Dl = size(mps[j], 1)
        Renv[j+1] = Matrix{T}(I, Dl, Dl)
    end
    return Lenv, Renv
end

function _norm_envs_for_train(mps::Vector{Array{T,3}}, ws::TrainWorkspace{T}) where {T<:Real}
    if ws.gram === nothing
        return norm_environments(mps)
    end
    if T === Float32
        _fill_gram_norm_env_float64!(mps, ws)
        return ws.env_L64, ws.env_R64
    end
    return norm_environments(mps, ws.gram::AbstractMatrix{T}, ws.gtmp_dd, ws.gmid_dr, ws.gacc_dl)
end

function _refresh_norm_envs_after_bond_train!(
    mps::Vector{Array{T,3}}, Lenv, Renv, j::Int, ws::TrainWorkspace{T},
) where {T<:Real}
    if ws.gram === nothing
        refresh_norm_envs_after_bond!(mps, Lenv, Renv, j)
    elseif T === Float32
        refresh_norm_envs_after_bond!(mps, Lenv, Renv, j, ws.gram::AbstractMatrix{Float32})
    else
        refresh_norm_envs_after_bond!(mps, Lenv, Renv, j, ws.gram::AbstractMatrix{T},
                                       ws.gtmp_dd, ws.gmid_dr, ws.gacc_dl)
    end
    return nothing
end

# ─── Training loop ────────────────────────────────────────────────────────────

function _train_log_flush!()
    flush(stdout); flush(stderr)
    if isdefined(Main, :IJulia)
        try; Main.IJulia.flush_all(); catch; end
    end
    return nothing
end

"""
    cosine_lr(epoch, n_epochs, η_max; η_min = η_max / 50) -> η_t

Cosine-annealing learning-rate schedule.  Returns the learning rate for
`epoch` (1-indexed) given a total of `n_epochs` epochs.

    η(t) = η_min + ½(η_max − η_min)(1 + cos(π(t−1)/n_epochs))

At `epoch = 1` the output equals `η_max`; at `epoch = n_epochs` it equals
approximately `η_min`.  Pass as `lr_schedule` to `train_mps!`:

    train_mps!(...; lr_schedule = cosine_lr)

or with a custom η_min via a closure:

    train_mps!(...; lr_schedule = (e, ne, η) -> cosine_lr(e, ne, η; η_min = 1e-5))
"""
function cosine_lr(epoch::Int, n_epochs::Int, η_max::Real; η_min::Real = η_max / 50)
    η_min + 0.5 * (η_max - η_min) * (1 + cos(π * (epoch - 1) / n_epochs))
end

"""
    train_mps!(mps, xi_data, n_epochs, η, D_max, ε_cut; ...) -> nll_hist

Full DMRG-style Born-machine training loop.

Each epoch performs:
1. Forward bond sweep (left → right) with incremental left boundary carry.
2. Backward bond sweep (right → left).
3. Left canonicalisation.
4. NLL estimate on `nll_samples` random training rows.

# Arguments
- `mps`            — MPS to train (mutated in place).
- `xi_data`        — `(N, M)` integer matrix of encoded paths.
- `n_epochs`       — number of full forward+backward sweep epochs.
- `η`              — Adam learning rate (base value passed to `lr_schedule`).
- `D_max`          — maximum bond dimension.
- `ε_cut`          — singular-value truncation floor (0 = no floor).

# Keyword arguments
- `feature_phi`     — `K × d` feature matrix (activates Gram inner product).
- `checkpoint_dir`  — directory for JLD2 checkpoints (nothing = no saving).
- `checkpoint_every`, `checkpoint_at` — epoch cadences.
- `bond_log`        — pre-allocated `Vector` to accumulate bond-spectrum records.
- `verbose`, `bond_progress`, `nll_samples` — logging controls.
- `lr_schedule`     — `f(epoch, n_epochs, η) -> η_t` called once per epoch.
                      `nothing` = constant learning rate.
- `val_data`        — held-out `(N_val, M)` integer matrix for validation NLL.
                      When provided, validation NLL is computed each epoch.
- `val_samples`     — number of rows to sample for the validation NLL estimate.
- `patience`        — early-stopping: stop after this many consecutive epochs
                      with no improvement in validation NLL. Ignored when
                      `val_data === nothing`.
- `val_nll_log`     — pre-allocated `Vector` appended with per-epoch val NLL.
"""
function train_mps!(
    mps::Vector{Array{T,3}}, xi_data, n_epochs, η, D_max, ε_cut;
    verbose            = true,
    bond_progress      = false,
    nll_samples        = 500,
    checkpoint_dir     = nothing,
    checkpoint_every   = nothing,
    checkpoint_at      = Int[],
    save_final         = true,
    checkpoint_meta    = nothing,
    bond_log           = nothing,
    feature_phi::Union{Nothing, AbstractMatrix} = nothing,
    lr_schedule::Union{Nothing, Function}       = nothing,
    val_data                                    = nothing,
    val_samples::Int                            = 2_000,
    patience::Int                               = typemax(Int),
    val_nll_log                                 = nothing,
) where {T<:Real}
    Ml     = length(mps)
    Nd     = size(xi_data, 1)
    d_loc  = size(mps[1], 2)
    _validate_training_inputs!(mps, xi_data; feature_phi = feature_phi, D_max = D_max)
    nll_hist   = Float64[]
    adam_state = AdamDict{T}()
    ws         = TrainWorkspace(T, Nd, d_loc, D_max; feature_phi = feature_phi)

    # Early-stopping state (only active when val_data is provided)
    best_val_nll      = Inf
    patience_counter  = 0
    stop_early        = false

    meta_base = Dict{String,Any}(
        "M" => Ml, "d" => d_loc, "eta" => η, "D_max" => D_max,
        "eps_cut" => ε_cut, "n_epochs" => n_epochs, "Nd_train" => Nd,
        "eltype" => string(T),
        "feature_phi" => feature_phi === nothing ? false : true,
    )
    if checkpoint_meta !== nothing
        for (k, v) in pairs(checkpoint_meta); meta_base[string(k)] = v; end
    end

    if verbose
        fm = ws.feature_phi === nothing ? "one-hot" : "feature_map (Gram)"
        println("train_mps!: Ml=", Ml, ", Nd=", Nd, ", d=", d_loc,
                ", D_max=", D_max, ", epochs=", n_epochs, ", ", fm)
        _train_log_flush!()
    end

    function do_checkpoint(epoch::Int)
        checkpoint_dir === nothing && return
        isdir(checkpoint_dir) || mkpath(checkpoint_dir)
        save_it = false
        if checkpoint_every !== nothing && checkpoint_every > 0 && epoch % checkpoint_every == 0
            save_it = true
        end
        epoch ∈ checkpoint_at && (save_it = true)
        save_it || return
        fn = joinpath(checkpoint_dir, "mps_epoch_$(lpad(epoch, 4, '0')).jld2")
        save_mps_bundle(fn, mps, nll_hist, epoch, meta_base)
        save_mps_bundle(joinpath(checkpoint_dir, "mps_latest.jld2"), mps, nll_hist, epoch, meta_base)
        verbose && (println("  → checkpoint: ", fn); _train_log_flush!())
    end

    for epoch in 1:n_epochs
        t_epoch = time()
        verbose && (println("— Epoch ", epoch, "/", n_epochs, " —"); _train_log_flush!())

        # Per-epoch learning rate (cosine annealing or constant)
        η_t = T(lr_schedule === nothing ? η : lr_schedule(epoch, n_epochs, η))
        # Right-canonicalize before the forward sweep so that every Renv[j+3] = I.
        # With left-canonical splits in the forward sweep, Lenv[j] = I as well, so
        # Z = ‖B‖²_F throughout — preventing Float32 overflow with large D_max.
        right_canonicalize_mps!(mps)

        # Use identity environments: the left transfer through right-canonical sites
        # can scale by up to Dl=D_max per step (Lenv would reach D_max^M ≈ 10^63,
        # overflowing Float32). Instead, start both envs as I and grow them
        # incrementally via _refresh_norm_envs_after_bond_train! during the sweep.
        if ws.gram === nothing
            Lenv, Renv = _canonical_envs_identity(mps)
        else
            Lenv, Renv = _norm_envs_for_train(mps, ws)   # Float64, no overflow
        end
        verbose && (println("  ↳ norm envs ready → forward sweep (", Ml - 1, " bonds) …"); _train_log_flush!())

        # Forward sweep (left → right)
        Lv = ones(T, Nd, 1)
        for j in 1:(Ml-1)
            bond_progress && verbose && (println("  · forward bond ", j, "/", Ml - 1); _train_log_flush!())
            update_pair!(ws, mps, xi_data, j, η_t, D_max, ε_cut, Lenv, Renv, adam_state;
                         Lv_carry = Lv, bond_log = bond_log,
                         epoch = epoch, sweep = :forward, d_phys = d_loc)
            _refresh_norm_envs_after_bond_train!(mps, Lenv, Renv, j, ws)
            Lv = extend_lv_after_bond!(ws, mps, xi_data, j, Lv)
        end

        verbose && (println("  ↳ forward done → canonicalize + rebuild envs + backward sweep …"); _train_log_flush!())

        # Normalize after forward sweep, then rebuild environments.
        # left_canonicalize_mps! makes Lenv[j] = I for all j (products of
        # left-isometric transfer matrices are identity), so the backward sweep
        # has Lenv = I at every active bond, mirroring the forward sweep guarantee.
        left_canonicalize_mps!(mps)

        # Pre-compute left boundary vector cache for the backward sweep.
        # Key invariant: backward bond j only modifies sites j and j+1, so sites
        # 1..j-1 are untouched when lv_cache[j] is consumed, making every cached
        # entry valid.  One O(M·Nd·D²) forward pass here replaces O(M²·Nd·D²) worth
        # of lv_prefix calls inside the backward loop.
        lv_cache = Vector{Matrix{T}}(undef, Ml)
        lv_cache[1] = ones(T, Nd, 1)
        for k in 1:(Ml - 1)
            lv_cache[k + 1] = extend_lv_after_bond!(ws, mps, xi_data, k, lv_cache[k])
        end

        # Backward sweep (right → left)
        if ws.gram === nothing
            Lenv, Renv = _canonical_envs_identity(mps)
        else
            Lenv, Renv = _norm_envs_for_train(mps, ws)
        end
        for j in (Ml-1):-1:1
            bond_progress && verbose && (println("  · backward bond ", j, "/", Ml - 1); _train_log_flush!())
            update_pair!(ws, mps, xi_data, j, η_t, D_max, ε_cut, Lenv, Renv, adam_state;
                         Lv_carry = lv_cache[j], bond_log = bond_log,
                         epoch = epoch, sweep = :backward, d_phys = d_loc)
            _refresh_norm_envs_after_bond_train!(mps, Lenv, Renv, j, ws)
        end

        verbose && (println("  ↳ backward done → canonicalize + NLL estimate …"); _train_log_flush!())

        left_canonicalize_mps!(mps)
        Le, _ = _norm_envs_for_train(mps, ws)
        Z_est = Float64(Le[Ml+1][1, 1])
        logZ  = log(max(Z_est, 1e-30))
        idx   = randperm(Nd)[1:min(nll_samples, Nd)]
        nll_atomic = Threads.Atomic{Float64}(0.0)
        Threads.@threads for k in 1:length(idx)
            i  = idx[k]
            p  = mps_amplitude(mps, xi_data[i, :]; phi = ws.feature_phi)
            Threads.atomic_add!(nll_atomic, logZ - 2.0 * log(max(abs(Float64(p)), 1e-30)))
        end
        nll   = nll_atomic[] / length(idx)
        push!(nll_hist, nll)
        bonds = join([size(mps[j], 3) for j in 1:Ml-1], ",")
        if verbose
            elapsed = round(time() - t_epoch; digits=3)
            println("Epoch $epoch/$n_epochs | NLL ≈ $(round(nll; digits=4)) | η=$(round(Float64(η_t); sigdigits=3)) | bonds=[$bonds] | $(elapsed) s")
            _train_log_flush!()
        end

        # ── Validation NLL + early stopping ───────────────────────────────────
        if val_data !== nothing
            Nv   = size(val_data, 1)
            vidx = randperm(Nv)[1:min(val_samples, Nv)]
            val_atomic = Threads.Atomic{Float64}(0.0)
            Threads.@threads for k in 1:length(vidx)
                i = vidx[k]
                p = mps_amplitude(mps, val_data[i, :]; phi = ws.feature_phi)
                Threads.atomic_add!(val_atomic, logZ - 2.0 * log(max(abs(Float64(p)), 1e-30)))
            end
            val_nll = val_atomic[] / length(vidx)
            val_nll_log !== nothing && push!(val_nll_log, val_nll)
            verbose && (println("  ↳ val NLL ≈ $(round(val_nll; digits=4))  (patience $patience_counter/$patience)"); _train_log_flush!())

            if val_nll < best_val_nll
                best_val_nll     = val_nll
                patience_counter = 0
            else
                patience_counter += 1
                if patience_counter >= patience
                    verbose && (println("Early stopping at epoch $epoch (no val improvement for $patience epochs)"); _train_log_flush!())
                    do_checkpoint(epoch)
                    stop_early = true
                end
            end
        end

        do_checkpoint(epoch)
        stop_early && break
    end

    if save_final && checkpoint_dir !== nothing
        isdir(checkpoint_dir) || mkpath(checkpoint_dir)
        fn = joinpath(checkpoint_dir, "mps_final.jld2")
        save_mps_bundle(fn, mps, nll_hist, n_epochs, meta_base; bond_log = bond_log)
        verbose && (println("→ final model: ", fn); _train_log_flush!())
    end
    return nll_hist
end
