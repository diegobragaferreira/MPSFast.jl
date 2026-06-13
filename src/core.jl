# core.jl — MPS gauge, amplitude, transfer matrices, norm environments.
#
# Conventions: A^j[α, σ, β]  ← (D_left, d, D_right), Float32 by default.
# Physical indices are 1-based: σ ∈ 1:d.

# ─── MPS initialisation ──────────────────────────────────────────────────────

"""
    init_mps(M, d, D_max; T = Float32, rng = Random.default_rng()) -> mps

Allocate a random left-canonical MPS of length `M`, physical dimension `d`,
and maximum bond dimension `D_max`. Bond dimensions grow as `min(d^j, D_max)`.

`T` controls element type (`Float32` recommended; `Float64` for diagnostics).
The MPS is immediately left-canonicalised after initialisation.
"""
function init_mps(
    M::Int, d::Int, D_max::Int;
    T::Type{<:Real} = Float32,
    rng::AbstractRNG = Random.default_rng(),
)
    # Build bond dimensions with a symmetric taper that never overflows:
    # grow from the left and right simultaneously, capping at D_max.
    function _safe_bond(k)
        lk = min(k, M - k)  # distance from the nearer end
        b = 1
        for _ in 1:lk
            b = min(b * d, D_max)
        end
        return b
    end
    Ds  = [_safe_bond(k) for k in 0:M]
    mps = [randn(rng, T, Ds[j], d, Ds[j+1]) for j in 1:M]
    left_canonicalize_mps!(mps)
    return mps
end

# ─── Amplitude ───────────────────────────────────────────────────────────────

"""
    mps_amplitude(mps, x; phi = nothing) -> scalar

Single-sample MPS amplitude `Ψ(x)` for an integer index vector `x[j]`.

When `phi === nothing`, `x[j]` indexes the physical leg directly (`K = d`).
When `phi` is a `K × d` feature matrix, each step contracts via
`M_{x[j]} = Σ_σ phi[x[j],σ] A[:,σ,:]` (Gram-weighted Hilbert space).

For `Float32` MPS with a feature map the contraction is promoted to
`Float64` internally for numerical stability.
"""
function mps_amplitude(
    mps::AbstractVector{<:AbstractArray{T,3}},
    x::AbstractVector{<:Integer};
    phi::Union{Nothing, AbstractMatrix} = nothing,
) where {T<:Real}
    Ml = length(mps)
    if phi === nothing
        @inbounds v = mps[1][1, x[1], :]'
        @inbounds for j in 2:Ml
            v = v * mps[j][:, x[j], :]
        end
        return v[1, 1]
    end
    # Float32 cores + Φ: accumulate amplitude in Float64 for stability.
    if T === Float32
        Tf = Float64
        Phi0 = phi isa Matrix{T} ? phi : Matrix{T}(phi)
        PhiF = Matrix{Tf}(Phi0)
        K, dphi = size(PhiF)
        @assert size(mps[1], 2) == dphi
        Aj = mps[1]
        Dl, d, Dr = size(Aj)
        @assert Dl == 1
        k1 = x[1]
        @assert 1 ≤ k1 ≤ K
        v = zeros(Tf, 1, Dr)
        @inbounds for σ in 1:d
            ph = PhiF[k1, σ]
            iszero(ph) && continue
            @simd for β in 1:Dr
                v[1, β] += ph * Tf(Aj[1, σ, β])
            end
        end
        @inbounds for j in 2:Ml
            Aj = mps[j]
            Dl, d, Dr = size(Aj)
            kj = x[j]
            @assert 1 ≤ kj ≤ K
            v2 = zeros(Tf, 1, Dr)
            @inbounds for σ in 1:d
                ph = PhiF[kj, σ]
                iszero(ph) && continue
                @inbounds for β in 1:Dr
                    s = zero(Tf)
                    @simd for α in 1:Dl
                        s += v[1, α] * Tf(Aj[α, σ, β])
                    end
                    v2[1, β] += ph * s
                end
            end
            v = v2
        end
        return T(v[1, 1])
    end
    Phi = phi isa Matrix{T} ? phi : Matrix{T}(phi)
    K, dphi = size(Phi)
    @assert size(mps[1], 2) == dphi
    Aj = mps[1]
    Dl, d, Dr = size(Aj)
    @assert Dl == 1
    k1 = x[1]
    @assert 1 ≤ k1 ≤ K
    v = zeros(T, 1, Dr)
    @inbounds for σ in 1:d
        ph = Phi[k1, σ]
        iszero(ph) && continue
        @simd for β in 1:Dr
            v[1, β] += ph * Aj[1, σ, β]
        end
    end
    @inbounds for j in 2:Ml
        Aj = mps[j]
        Dl, d, Dr = size(Aj)
        kj = x[j]
        @assert 1 ≤ kj ≤ K
        v2 = zeros(T, 1, Dr)
        @inbounds for σ in 1:d
            ph = Phi[kj, σ]
            iszero(ph) && continue
            @inbounds for β in 1:Dr
                s = zero(T)
                @simd for α in 1:Dl
                    s += v[1, α] * Aj[α, σ, β]
                end
                v2[1, β] += ph * s
            end
        end
        v = v2
    end
    return v[1, 1]
end

# ─── Classification helpers ───────────────────────────────────────────────────

"""
    class_probabilities(mps, xi_path, n_classes; phi = nothing) -> Vector{Float64}

Born-rule class probabilities `p(y=c | xi_path)` for a fixed encoded path
`xi_path` (length `length(mps)-1`). Enumerates `c = 1:n_classes`, sets the
label site to `c`, and normalises `|Ψ(xi_path, c)|²`.

Pass `phi = nothing` for basis/binary path + one-hot label (recommended).
"""
function class_probabilities(
    mps::AbstractVector{<:AbstractArray{T,3}},
    xi_path::AbstractVector{<:Integer},
    n_classes::Int;
    phi::Union{Nothing, AbstractMatrix} = nothing,
) where {T<:Real}
    Ml = length(mps)
    @assert length(xi_path) == Ml - 1
    amps = Vector{Float64}(undef, n_classes)
    @inbounds for c in 1:n_classes
        x = vcat(collect(Int, xi_path), c)
        amps[c] = abs2(Float64(mps_amplitude(mps, x; phi = phi)))
    end
    s = sum(amps)
    !(s > 0) || !isfinite(s) && return fill(1.0 / n_classes, n_classes)
    return amps ./ s
end

"""Argmax of `class_probabilities` (1-based class index)."""
function predict_class(
    mps::AbstractVector{<:AbstractArray{T,3}},
    xi_path::AbstractVector{<:Integer},
    n_classes::Int;
    phi::Union{Nothing, AbstractMatrix} = nothing,
) where {T<:Real}
    return argmax(class_probabilities(mps, xi_path, n_classes; phi = phi))
end

"""Fraction of correctly classified rows in `xi_data` (label in last column)."""
function classification_accuracy(
    mps::AbstractVector{<:AbstractArray{T,3}},
    xi_data::AbstractMatrix{<:Integer},
    n_classes::Int;
    phi::Union{Nothing, AbstractMatrix} = nothing,
) where {T<:Real}
    Nd = size(xi_data, 1)
    correct = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:Nd
        xi_row = xi_data[i, :]
        y_true = Int(xi_row[end])
        y_hat  = predict_class(mps, xi_row[1:end-1], n_classes; phi = phi)
        y_hat == y_true && Threads.atomic_add!(correct, 1)
    end
    return correct[] / Nd
end

# ─── Gauge ────────────────────────────────────────────────────────────────────

"""Left-canonicalise (QR sweep, in-place). Last tensor carries the norm."""
function left_canonicalize_mps!(mps::Vector{Array{T,3}}) where {T<:Real}
    M = length(mps)
    M == 1 && (mps[1] ./= norm(mps[1]); return mps)
    # Carry the entire sweep in Float64 to prevent Float32 overflow.
    # For a fresh randn-initialised MPS, ‖mps[j]‖_F ≈ √(Dl·d·Dr) ≈ 600 for
    # D_max=150.  After M−1 QR/carry steps the accumulated intermediate norms
    # reach ~600^28 ≈ 10^77, far above Float32 max (3.4e38) but safely within
    # Float64 (1.8e308).  After left-canonicalisation every mps[j<M] is
    # left-isometric (‖·‖_F = √Dr ≤ √D_max ≈ 12), so converting back to T is safe.
    mps64 = [Float64.(t) for t in mps]
    @inbounds for j in 1:M-1
        Dl, dj, Dr           = size(mps64[j])
        Dl_n, dj_n, Dr_n     = size(mps64[j+1])   # j+1 may have a different physical dim
        F = qr(reshape(mps64[j], Dl * dj, Dr))
        Q = Matrix(F.Q)
        k = size(Q, 2)
        mps64[j]   = reshape(Q, Dl, dj, k)
        Anext      = reshape(mps64[j+1], Dl_n, dj_n * Dr_n)
        mps64[j+1] = reshape(Matrix{Float64}(F.R) * Anext, k, dj_n, Dr_n)
    end
    mps64[M] ./= norm(mps64[M])
    for j in 1:M
        mps[j] = T.(mps64[j])
    end
    return mps
end

"""
    right_canonicalize_mps!(mps)

Right-to-left QR sweep that makes every site right-isometric:
    Σ_{β,σ} A[α,σ,β] · A[α',σ,β] = δ_{αα'}
i.e. rows of reshape(A, Dl, d·Dr) are orthonormal.

After this call, the right norm-environments satisfy Renv[j] = I for every j,
so the partition function at any active bond reduces to Z = ‖B‖²_F (Frobenius
norm of the merged tensor). This eliminates Float32 overflow/underflow for large
D_max/M combinations (the ITensor mixed-canonical-form approach).
"""
function right_canonicalize_mps!(mps::Vector{Array{T,3}}) where {T<:Real}
    M = length(mps)
    M == 1 && (mps[1] ./= norm(mps[1]); return mps)
    @inbounds for j in M:-1:2
        Dl, dj, Dr = size(mps[j])
        # A right-isometric site requires Dl ≤ dj·Dr (wide matrix) so that its
        # Dl rows can be made orthonormal.  Tall sites (Dl > dj·Dr) — e.g. a
        # classification label site where dj = n_classes << Dl — cannot satisfy
        # this; skip them rather than crashing inside `lq`.  Their norm is
        # absorbed into the next site encountered via the normal carry path,
        # and the training sweep's environment refresh will produce the correct
        # right-transfer matrix for those sites during the first bond update.
        if Dl > dj * Dr
            continue
        end
        # LQ of the (Dl × dj·Dr) matrix: N = L·Q, where L is (Dl×Dl) lower
        # triangular and Q is (Dl × dj·Dr) with Q·Q' = I (orthonormal rows).
        # mps[j] ← reshape(Q, Dl, dj, Dr)  →  right-isometric ✓
        # Carry L into the right bond of mps[j-1].
        N = reshape(mps[j], Dl, dj * Dr)
        F = lq(N)
        Q = Matrix(F.Q)              # (Dl × dj·Dr)
        L = Matrix(F.L)              # (Dl × Dl)
        mps[j] = reshape(Q, Dl, dj, Dr)
        Dl2, dj2, _ = size(mps[j-1])
        mps[j-1] = reshape(reshape(mps[j-1], Dl2 * dj2, Dl) * L, Dl2, dj2, Dl)
    end
    n1 = norm(mps[1])
    if iszero(n1) || !isfinite(n1)
        @warn "right_canonicalize_mps!: ||mps[1]|| = $n1 after carries — MPS may have been zero/NaN before canonicalization."
    else
        mps[1] ./= n1
    end
    return mps
end

# ─── Transfer matrices ────────────────────────────────────────────────────────

"""Left transfer: E_new[β,β'] = Σ_{α,α',σ} A[α,σ,β] E[α,α'] A[α',σ,β']."""
@inline function _transfer_left!(
    Enew::AbstractMatrix{T}, E::AbstractMatrix{T}, A::AbstractArray{T,3},
) where {T<:Real}
    Dl, d, Dr = size(A)
    Y = E * reshape(A, Dl, d * Dr)
    mul!(Enew, reshape(A, Dl * d, Dr)', reshape(Y, Dl * d, Dr))
    return Enew
end

"""Right transfer: E_new[α,α'] = Σ_{β,β',σ} A[α,σ,β] E[β,β'] A[α',σ,β']."""
@inline function _transfer_right!(
    Enew::AbstractMatrix{T}, E::AbstractMatrix{T}, A::AbstractArray{T,3},
) where {T<:Real}
    Dl, d, Dr = size(A)
    W = reshape(A, Dl * d, Dr) * E
    mul!(Enew, reshape(A, Dl, d * Dr), reshape(W, Dl, d * Dr)')
    return Enew
end

"""
Gram-weighted left transfer with metric G (d×d):

    E_new[β,β'] = Σ_{σ,τ} G[σ,τ] Σ_{α,α'} A[α,σ,β] E[α,α'] A[α',τ,β']

`Xbuf` and `Wbuf` are scratch matrices of size at least `D_max × D_max`.
"""
function _transfer_left_G!(
    Enew::AbstractMatrix{T}, E::AbstractMatrix{T},
    A::AbstractArray{T,3}, G::AbstractMatrix{T},
    Xbuf::AbstractMatrix{T}, Wbuf::AbstractMatrix{T},
) where {T<:Real}
    Dl, d, Dr = size(A)
    fill!(Enew, zero(T))
    X = view(Xbuf, 1:Dr, 1:Dl)
    W = view(Wbuf, 1:Dr, 1:Dr)
    @inbounds for σ in 1:d
        mul!(X, transpose(view(A, :, σ, :)), E)
        @inbounds for τ in 1:d
            g = G[σ, τ]
            iszero(g) && continue
            mul!(W, X, view(A, :, τ, :))
            axpy!(g, W, Enew)
        end
    end
    return Enew
end

"""Gram-weighted right transfer."""
function _transfer_right_G!(
    Enew::AbstractMatrix{T}, E::AbstractMatrix{T},
    A::AbstractArray{T,3}, G::AbstractMatrix{T},
    mid_dl_dr::AbstractMatrix{T}, acc_dl_dl::AbstractMatrix{T},
) where {T<:Real}
    Dl, d, Dr = size(A)
    fill!(Enew, zero(T))
    @inbounds for σ in 1:d
        @inbounds for τ in 1:d
            g = G[σ, τ]
            iszero(g) && continue
            mul!(view(mid_dl_dr, 1:Dl, 1:Dr), view(A, :, σ, :), E)
            mul!(view(acc_dl_dl, 1:Dl, 1:Dl),
                 view(mid_dl_dr, 1:Dl, 1:Dr),
                 transpose(view(A, :, τ, :)))
            @inbounds for j in 1:Dl, i in 1:Dl
                Enew[i, j] += g * acc_dl_dl[i, j]
            end
        end
    end
    return Enew
end

# ─── Norm environments ────────────────────────────────────────────────────────

"""Maximum bond dimension across all MPS cores."""
function _max_mps_bond(mps::Vector{<:AbstractArray{<:Real,3}})
    m = 1
    @inbounds for A in mps
        m = max(m, size(A, 1), size(A, 3))
    end
    return m
end

"""
    norm_environments(mps) -> (Lenv, Renv)

Left and right norm-environment chains for the standard Hilbert inner product.
`Lenv[j]` is the left environment arriving at site `j` (size `D_{j-1} × D_{j-1}`);
`Renv[j+2]` is the right environment leaving site `j`.
"""
function norm_environments(mps::Vector{Array{T,3}}) where {T<:Real}
    N = length(mps)
    Lenv = Vector{Matrix{T}}(undef, N + 1)
    Renv = Vector{Matrix{T}}(undef, N + 2)
    Lenv[1]   = Matrix{T}(I, 1, 1)
    Renv[N+2] = Matrix{T}(I, 1, 1)
    @inbounds for j in 1:N
        Dr = size(mps[j], 3)
        Lenv[j+1] = Matrix{T}(undef, Dr, Dr)
        _transfer_left!(Lenv[j+1], Lenv[j], mps[j])
    end
    @inbounds for j in N:-1:1
        Dl = size(mps[j], 1)
        Renv[j+1] = Matrix{T}(undef, Dl, Dl)
        _transfer_right!(Renv[j+1], Renv[j+2], mps[j])
    end
    return Lenv, Renv
end

"""
Gram norm-environment overload: uses metric `G = Φ'Φ` on every physical leg.
`tmp_dd`, `mid_dr`, `acc_dl` are scratch buffers of size `≥ D_max × D_max`.
"""
function norm_environments(
    mps::Vector{Array{T,3}},
    G::AbstractMatrix{T},
    tmp_dd::Matrix{T},
    mid_dr::Matrix{T},
    acc_dl::Matrix{T},
) where {T<:Real}
    N = length(mps)
    if T === Float32
        return _norm_environments_gram_f32_sweep(mps, G, size(tmp_dd, 1))
    end
    Lenv = Vector{Matrix{T}}(undef, N + 1)
    Renv = Vector{Matrix{T}}(undef, N + 2)
    Lenv[1]   = Matrix{T}(I, 1, 1)
    Renv[N+2] = Matrix{T}(I, 1, 1)
    @inbounds for j in 1:N
        Dr = size(mps[j], 3)
        Lenv[j+1] = Matrix{T}(undef, Dr, Dr)
        _transfer_left_G!(Lenv[j+1], Lenv[j], mps[j], G, tmp_dd, mid_dr)
    end
    @inbounds for j in N:-1:1
        Dl = size(mps[j], 1)
        Dr = size(mps[j], 3)
        Renv[j+1] = Matrix{T}(undef, Dl, Dl)
        _transfer_right_G!(Renv[j+1], Renv[j+2], mps[j], G,
                           view(mid_dr, 1:Dl, 1:Dr),
                           view(acc_dl, 1:Dl, 1:Dl))
    end
    return Lenv, Renv
end

"""Full Gram norm-environment sweep promoted to Float64 (Float32 MPS overflow guard)."""
function _norm_environments_gram_f32_sweep(
    mps::Vector{Array{Float32,3}}, G::AbstractMatrix{Float32}, Dmx::Int,
)
    N  = length(mps)
    Tf = Float64
    mF = [Tf.(A) for A in mps]
    GF = Matrix{Tf}(G)
    tX = Matrix{Tf}(undef, Dmx, Dmx)
    tW = Matrix{Tf}(undef, Dmx, Dmx)
    tM = Matrix{Tf}(undef, Dmx, Dmx)
    tA = Matrix{Tf}(undef, Dmx, Dmx)
    Lf = Vector{Matrix{Tf}}(undef, N + 1)
    Rf = Vector{Matrix{Tf}}(undef, N + 2)
    Lf[1]   = Matrix{Tf}(I, 1, 1)
    Rf[N+2] = Matrix{Tf}(I, 1, 1)
    @inbounds for j in 1:N
        Dr = size(mF[j], 3)
        Lf[j+1] = Matrix{Tf}(undef, Dr, Dr)
        _transfer_left_G!(Lf[j+1], Lf[j], mF[j], GF, tX, tW)
    end
    @inbounds for j in N:-1:1
        Dl = size(mF[j], 1)
        Dr = size(mF[j], 3)
        Rf[j+1] = Matrix{Tf}(undef, Dl, Dl)
        _transfer_right_G!(Rf[j+1], Rf[j+2], mF[j], GF,
                           view(tM, 1:Dl, 1:Dr),
                           view(tA, 1:Dl, 1:Dl))
    end
    return Lf, Rf
end

"""
    log_partition_function(mps; feature_phi = nothing) -> Float64

`log Z` where `Z = ‖Ψ‖²`. Uses the standard transfer norm when
`feature_phi === nothing`, or the Gram-weighted norm `G = Φ'Φ` otherwise.
"""
function log_partition_function(
    mps::Vector{Array{T,3}};
    feature_phi::Union{Nothing, AbstractMatrix} = nothing,
) where {T<:Real}
    Ml = length(mps)
    Dmx = _max_mps_bond(mps)
    if feature_phi === nothing
        L, _ = norm_environments(mps)
        z = Float64(L[Ml + 1][1, 1])
    else
        PhiM = feature_phi isa Matrix{T} ? feature_phi : Matrix{T}(feature_phi)
        G = PhiM' * PhiM
        if T === Float32
            Lf, _ = _norm_environments_gram_f32_sweep(mps, Matrix{Float32}(G), Dmx)
            z = Float64(Lf[Ml + 1][1, 1])
        else
            tmp = Matrix{T}(undef, Dmx, Dmx)
            L, _ = norm_environments(mps, Matrix{T}(G), tmp, tmp, tmp)
            z = Float64(L[Ml + 1][1, 1])
        end
    end
    return log(max(z, 1e-300))
end

# ─── Validation ───────────────────────────────────────────────────────────────

"""
    _validate_training_inputs!(mps, xi_data; feature_phi, D_max)

Check site count, physical dimension, bucket range and bond consistency before
training or sampling. Raises informative `AssertionError` on any mismatch.
"""
function _validate_training_inputs!(
    mps::Vector{Array{T,3}}, xi_data::AbstractMatrix{<:Integer};
    feature_phi::Union{Nothing, AbstractMatrix} = nothing,
    D_max::Union{Nothing, Int} = nothing,
) where {T<:Real}
    Ml = length(mps)
    Nd, Mxi = size(xi_data)
    @assert Ml == Mxi "xi_data has $Mxi columns but MPS has $Ml sites."
    @assert Nd >= 1   "xi_data must have at least one row."
    d = size(mps[1], 2)
    # Uniform physical dimension is required when a feature map is used (the Φ
    # matrix must cover every site).  For one-hot / classification MPS the last
    # (label) site may have a different physical dim, so only check when needed.
    if feature_phi !== nothing
        @inbounds for j in 2:Ml
            @assert size(mps[j], 2) == d "Physical dim mismatch at site $j."
        end
    end
    mxi, mx = extrema(xi_data)
    @assert mxi >= 1 "xi_data minimum is $mxi (bucket indices must be ≥ 1)."
    if feature_phi !== nothing
        K = size(feature_phi, 1)
        @assert mx <= K "xi_data max bucket $mx > Φ rows K=$K."
        @assert size(feature_phi, 2) == d "Φ columns ≠ MPS d=$d."
    end
    if D_max !== nothing
        Db = _max_mps_bond(mps)
        @assert Db <= D_max "MPS max bond $Db > D_max=$D_max."
    end
    @inbounds for j in 1:(Ml - 1)
        br = size(mps[j], 3)
        bl = size(mps[j + 1], 1)
        @assert br == bl "Bond mismatch $(j)→$(j+1): right $br ≠ left $bl."
    end
    return nothing
end

# ─── Norm environment refresh ─────────────────────────────────────────────────

"""After `mps[j]` and `mps[j+1]` change: rebuild `Lenv` rightward and `Renv` leftward."""
function refresh_norm_envs_after_bond!(
    mps::Vector{Array{T,3}},
    Lenv::Vector{Matrix{T}},
    Renv::Vector{Matrix{T}},
    j::Int,
) where {T<:Real}
    N = length(mps)
    @assert 1 <= j < N
    @inbounds for k in j:N
        Dr = size(mps[k], 3)
        Lenv[k+1] = Matrix{T}(undef, Dr, Dr)
        _transfer_left!(Lenv[k+1], Lenv[k], mps[k])
    end
    @inbounds for jj in (j+1):-1:1
        Dl = size(mps[jj], 1)
        Renv[jj+1] = Matrix{T}(undef, Dl, Dl)
        _transfer_right!(Renv[jj+1], Renv[jj+2], mps[jj])
    end
    return nothing
end

function refresh_norm_envs_after_bond!(
    mps::Vector{Array{Float32,3}},
    Lenv::Vector{Matrix{Float64}},
    Renv::Vector{Matrix{Float64}},
    j::Int,
    G::AbstractMatrix{Float32},
)
    N  = length(mps)
    @assert 1 <= j < N
    Tf = Float64
    mF = [Tf.(A) for A in mps]
    GF = Matrix{Tf}(G)
    Dmx = maximum(k -> max(size(mps[k], 1), size(mps[k], 3)), 1:N)
    tX = Matrix{Tf}(undef, Dmx, Dmx)
    tW = Matrix{Tf}(undef, Dmx, Dmx)
    tM = Matrix{Tf}(undef, Dmx, Dmx)
    tA = Matrix{Tf}(undef, Dmx, Dmx)
    Ej = Lenv[j]
    @inbounds for k in j:N
        Dr = size(mF[k], 3)
        Lnext = Matrix{Tf}(undef, Dr, Dr)
        _transfer_left_G!(Lnext, Ej, mF[k], GF, tX, tW)
        Lenv[k+1] = Lnext
        Ej = Lnext
    end
    Er = Renv[j+3]
    @inbounds for jj in (j+1):-1:1
        Dl  = size(mF[jj], 1)
        Drs = size(mF[jj], 3)
        Rnext = Matrix{Tf}(undef, Dl, Dl)
        _transfer_right_G!(Rnext, Er, mF[jj], GF,
                           view(tM, 1:Dl, 1:Drs),
                           view(tA, 1:Dl, 1:Dl))
        Renv[jj+1] = Rnext
        Er = Rnext
    end
    return nothing
end

function refresh_norm_envs_after_bond!(
    mps::Vector{Array{T,3}},
    Lenv::Vector{Matrix{T}},
    Renv::Vector{Matrix{T}},
    j::Int,
    G::AbstractMatrix{T},
    tmp_dd::Matrix{T},
    mid_dr::Matrix{T},
    acc_dl::Matrix{T},
) where {T<:Real}
    N = length(mps)
    @assert 1 <= j < N
    @inbounds for k in j:N
        Dr = size(mps[k], 3)
        Lenv[k+1] = Matrix{T}(undef, Dr, Dr)
        _transfer_left_G!(Lenv[k+1], Lenv[k], mps[k], G, tmp_dd, mid_dr)
    end
    @inbounds for jj in (j+1):-1:1
        Dl  = size(mps[jj], 1)
        Drs = size(mps[jj], 3)
        Renv[jj+1] = Matrix{T}(undef, Dl, Dl)
        _transfer_right_G!(Renv[jj+1], Renv[jj+2], mps[jj], G,
                           view(mid_dr, 1:Dl, 1:Drs),
                           view(acc_dl, 1:Dl, 1:Dl))
    end
    return nothing
end
