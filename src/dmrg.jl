# dmrg.jl — Genuine two-site (and excited-state) DMRG for a Hamiltonian given
# as an MPO, applied to the market-making Hamiltonian of `market_making.jl`
# (Coutinho report, §12 milestone 2: "implement two-site DMRG and excited-state
# DMRG, including residual and quote metrics" / "reproduce table 1 without
# forming the full state vector during optimization").
#
# Distinct from `training.jl`'s Born-machine "DMRG-style" NLL-gradient sweep:
# this is a real variational ground-state eigensolver — at each bond, the
# local 2-site tensor is updated by *diagonalizing* the effective Hamiltonian
# (matrix-free Lanczos via KrylovKit) rather than by gradient descent on data.
#
# All contractions use `TensorOperations.@tensor` (not manual reshape/
# permutedims) to avoid silent index-ordering bugs.

using TensorOperations
using KrylovKit: eigsolve
using LinearAlgebra
using Random

# ── Initialization ──────────────────────────────────────────────────────────

"""
    random_mps_hetero(dims, maxdim; rng, T=Float64) -> mps

Random left-canonical MPS with per-site physical dimensions `dims` (may be
heterogeneous, unlike `MPSFast.init_mps`) and maximum bond dimension `maxdim`.
"""
function random_mps_hetero(
    dims::Vector{Int}, maxdim::Int;
    rng::AbstractRNG=Random.default_rng(), T::Type{<:Real}=Float64,
)
    N = length(dims)
    Ds = Vector{Int}(undef, N + 1)
    Ds[1] = 1
    Ds[N + 1] = 1
    for j in 1:N
        left = min(Ds[j] * dims[j], maxdim)
        Ds[j + 1] = j == N ? 1 : left
    end
    # symmetric taper from both ends (mirrors MPSFast.init_mps)
    for j in 1:N
        from_left = 1
        for i in 1:j
            from_left = min(from_left * dims[i], maxdim)
        end
        from_right = 1
        for i in N:-1:(j + 1)
            from_right = min(from_right * dims[i], maxdim)
        end
        Ds[j + 1] = min(from_left, from_right)
    end
    mps = [randn(rng, T, Ds[j], dims[j], Ds[j + 1]) for j in 1:N]
    left_canonicalize_mps!(mps)
    return mps
end

# ── Inner products and MPO application (all via @tensor) ──────────────────────

"""`⟨A|B⟩` for two MPS with identical physical dimensions but possibly
different bond dimensions."""
function mps_inner_product(A::Vector{<:Array{Float64,3}}, B::Vector{<:Array{Float64,3}})
    N = length(A)
    @assert length(B) == N
    E = ones(Float64, 1, 1)
    @inbounds for j in 1:N
        @tensor Enew[bp, b] := E[ap, a] * A[j][ap, s, bp] * B[j][a, s, b]
        E = Enew
    end
    @assert size(E) == (1, 1)
    return E[1, 1]
end

"""
    apply_mpo_to_mps(mps, mpo) -> new_mps

Exact (uncompressed) application of an MPO to an MPS: site `j` gets bond
dimensions `(Dl*Rl, Dr*Rr)`. Physical dimensions are unchanged.
"""
function apply_mpo_to_mps(mps::Vector{<:Array{Float64,3}}, mpo::Vector{<:Array{Float64,4}})
    N = length(mps)
    @assert length(mpo) == N
    out = Vector{Array{Float64,3}}(undef, N)
    @inbounds for j in 1:N
        A = mps[j]
        W = mpo[j]
        Da, d, Db = size(A)
        Rm, dout, din, Rn = size(W)
        @assert d == din
        @tensor T[a, m, sp, b, n] := A[a, s, b] * W[m, sp, s, n]
        out[j] = reshape(T, Da * Rm, dout, Db * Rn)
    end
    return out
end

"""`‖H|ψ⟩ − E|ψ⟩‖₂` (Ritz residual) for a normalised MPS `ψ` with reported
Rayleigh-quotient energy `E`, computed exactly via `apply_mpo_to_mps` +
`norm_environments` (no dense vector is ever formed)."""
function dmrg_residual(mps::Vector{<:Array{Float64,3}}, mpo::Vector{<:Array{Float64,4}}, E::Real)
    hpsi = apply_mpo_to_mps(mps, mpo)
    hpsi_f = [Array{Float64,3}(A) for A in hpsi]
    hh = mps_inner_product(hpsi_f, hpsi_f)          # ⟨Hψ|Hψ⟩ = ⟨ψ|H²|ψ⟩
    return sqrt(max(hh - E^2, 0.0))
end

"""Rayleigh quotient `⟨ψ|H|ψ⟩` for a normalised MPS via exact MPO application."""
function dmrg_energy(mps::Vector{<:Array{Float64,3}}, mpo::Vector{<:Array{Float64,4}})
    hpsi = apply_mpo_to_mps(mps, mpo)
    mps_f = [Array{Float64,3}(A) for A in mps]
    hpsi_f = [Array{Float64,3}(A) for A in hpsi]
    return mps_inner_product(mps_f, hpsi_f)
end

# ── MPO environments ────────────────────────────────────────────────────────

"""Left environment update: `Lnew[b',m',b] = Σ Lenv[a',m,a] A[a',s',b'] W[m,s',s,m'] A[a,s,b]`."""
function _update_Lenv(Lenv::Array{Float64,3}, A::Array{Float64,3}, W::Array{Float64,4})
    @tensor Lnew[bp, mp, b] := Lenv[ap, m, a] * A[ap, sp, bp] * W[m, sp, s, mp] * A[a, s, b]
    return Lnew
end

"""Right environment update: `Rnew[a',m',a] = Σ Renv[b',m,b] A[a',s',b'] W[m',s',s,m] A[a,s,b]`."""
function _update_Renv(Renv::Array{Float64,3}, A::Array{Float64,3}, W::Array{Float64,4})
    @tensor Rnew[ap, mp, a] := Renv[bp, m, b] * A[ap, sp, bp] * W[mp, sp, s, m] * A[a, s, b]
    return Rnew
end

"""Full left/right MPO-environment chains for `mps` under `mpo` (report §6.2 style)."""
function mpo_environments(mps::Vector{<:Array{Float64,3}}, mpo::Vector{<:Array{Float64,4}})
    N = length(mps)
    Lenv = Vector{Array{Float64,3}}(undef, N + 1)
    Renv = Vector{Array{Float64,3}}(undef, N + 2)
    Lenv[1] = ones(Float64, 1, 1, 1)
    Renv[N + 2] = ones(Float64, 1, 1, 1)
    @inbounds for j in 1:N
        Lenv[j + 1] = _update_Lenv(Lenv[j], Array{Float64,3}(mps[j]), Array{Float64,4}(mpo[j]))
    end
    @inbounds for j in N:-1:1
        Renv[j + 1] = _update_Renv(Renv[j + 2], Array{Float64,3}(mps[j]), Array{Float64,4}(mpo[j]))
    end
    return Lenv, Renv
end

# ── Two-site effective Hamiltonian (matrix-free) ───────────────────────────

"""
    apply_Heff_twosite(Lenv, W1, W2, Renv, psi) -> out

Effective two-site Hamiltonian acting on `psi[a,s1,s2,b]` (report §6.2):
`out[a,s1',s2',b] = Σ Lenv[a,m0,a2] W1[m0,s1',s1,m1] W2[m1,s2',s2,m2] Renv[b,m2,b2] psi[a2,s1,s2,b2]`.

Implemented as four sequential pairwise contractions (one MPO layer /
environment at a time) rather than a single 5-operand `@tensor` expression:
each pairwise step is an unambiguous generalized matrix multiply, which both
avoids relying on the contraction-order optimizer and keeps the cost
polynomial (`O(χ³dR + χ²d³R²)`-ish) instead of the much worse scaling a naive
contraction order can fall into for large bond dimension χ.
"""
function apply_Heff_twosite(
    Lenv::Array{Float64,3}, W1::Array{Float64,4}, W2::Array{Float64,4},
    Renv::Array{Float64,3}, psi::Array{Float64,4},
)
    @tensor tmp1[a, m0, s1, s2, b] := Lenv[a, m0, a2] * psi[a2, s1, s2, b]
    @tensor tmp2[a, s1p, m1, s2, b] := tmp1[a, m0, s1, s2, b] * W1[m0, s1p, s1, m1]
    @tensor tmp3[a, s1p, s2p, m2, b] := tmp2[a, s1p, m1, s2, b] * W2[m1, s2p, s2, m2]
    @tensor out[a, s1p, s2p, bp] := tmp3[a, s1p, s2p, m2, b] * Renv[bp, m2, b]
    return out
end

"""Merge two adjacent MPS cores into a single two-site tensor `(Dl,d1,d2,Dr)`."""
function merge_two_site(A1::Array{Float64,3}, A2::Array{Float64,3})
    @tensor psi[a, s1, s2, b] := A1[a, s1, m] * A2[m, s2, b]
    return psi
end

"""Truncated SVD split of a two-site tensor back into `(A1,A2)` with bond ≤ `maxdim`.
`direction = :right` folds singular values into `A2` (canonical center moves right);
`:left` folds into `A1` (center moves left)."""
function split_two_site(psi::Array{Float64,4}, maxdim::Int, direction::Symbol; cutoff::Float64=0.0)
    Dl, d1, d2, Dr = size(psi)
    M = reshape(psi, Dl * d1, d2 * Dr)
    F = svd(M)
    r = min(length(F.S), maxdim)
    # discard negligible singular values (relative cutoff)
    if cutoff > 0 && r > 1
        thresh = cutoff * F.S[1]
        r = max(1, count(>=(thresh), F.S[1:r]))
    end
    U = F.U[:, 1:r]
    S = F.S[1:r]
    Vt = F.Vt[1:r, :]
    if direction == :right
        A1 = reshape(U, Dl, d1, r)
        A2 = reshape(Diagonal(S) * Vt, r, d2, Dr)
    elseif direction == :left
        A1 = reshape(U * Diagonal(S), Dl, d1, r)
        A2 = reshape(Vt, r, d2, Dr)
    else
        error("direction must be :left or :right")
    end
    return A1, A2, S
end

# ── Two-site DMRG sweep ─────────────────────────────────────────────────────

"""
    dmrg_ground_state(mpo; maxdim=32, n_sweeps=10, cutoff=1e-14,
                       rng=Random.default_rng(), mps0=nothing, verbose=false)
        -> (E, mps, energy_history)

Variational two-site ground-state search (report §6.2, §12 milestone 2):
at each bond, the effective two-site Hamiltonian (built from MPO environments)
is diagonalized via matrix-free Lanczos (`KrylovKit.eigsolve`); the result is
truncated back to `maxdim` via SVD. Never forms the full `∏dᵢ`-dimensional
state vector.
"""
function dmrg_ground_state(
    mpo::Vector{<:Array{Float64,4}};
    maxdim::Int=32, n_sweeps::Int=10, cutoff::Float64=1e-14,
    rng::AbstractRNG=Random.default_rng(),
    mps0::Union{Nothing,Vector{<:Array{Float64,3}}}=nothing,
    krylovdim::Int=12, krylov_tol::Float64=1e-10,
    verbose::Bool=false,
)
    N = length(mpo)
    dims = [size(W, 2) for W in mpo]
    mps = mps0 === nothing ? random_mps_hetero(dims, maxdim; rng=rng) :
          [Array{Float64,3}(A) for A in mps0]

    # Start in right-canonical form so the initial Lenv/Renv chain is trivial to build.
    right_canonicalize_mps!(mps)
    Lenv, Renv = mpo_environments(mps, mpo)

    energy_history = Float64[]
    E = NaN
    for sweep in 1:n_sweeps
        # ---- left-to-right half sweep (center moves right) ----
        for j in 1:(N - 1)
            psi0 = merge_two_site(mps[j], mps[j + 1])
            sz = length(psi0)
            f = x -> vec(apply_Heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], reshape(x, size(psi0))))
            vals, vecs, info = eigsolve(f, vec(psi0), 1, :SR; ishermitian=true,
                                        tol=krylov_tol, krylovdim=krylovdim, maxiter=100)
            E = vals[1]
            psi_opt = reshape(vecs[1], size(psi0))
            A1, A2, _ = split_two_site(psi_opt, maxdim, :right; cutoff=cutoff)
            mps[j], mps[j + 1] = A1, A2
            Lenv[j + 1] = _update_Lenv(Lenv[j], mps[j], mpo[j])
            push!(energy_history, E)
            verbose && println("  sweep $sweep  bond $j→$(j+1) (L→R)  E = $E  χ = $(size(A2,1))")
        end
        # ---- right-to-left half sweep (center moves left) ----
        for j in (N - 1):-1:1
            psi0 = merge_two_site(mps[j], mps[j + 1])
            f = x -> vec(apply_Heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], reshape(x, size(psi0))))
            vals, vecs, info = eigsolve(f, vec(psi0), 1, :SR; ishermitian=true,
                                        tol=krylov_tol, krylovdim=krylovdim, maxiter=100)
            E = vals[1]
            psi_opt = reshape(vecs[1], size(psi0))
            A1, A2, _ = split_two_site(psi_opt, maxdim, :left; cutoff=cutoff)
            mps[j], mps[j + 1] = A1, A2
            Renv[j + 2] = _update_Renv(Renv[j + 3], mps[j + 1], mpo[j + 1])
            push!(energy_history, E)
            verbose && println("  sweep $sweep  bond $(j+1)→$j (R→L)  E = $E  χ = $(size(A1,3))")
        end
    end
    return E, mps, energy_history
end

# ── Excited-state DMRG via ground-state deflation ──────────────────────────

"""
    dmrg_excited_state(mpo, ground_mps; maxdim=32, n_sweeps=10, weight=..., kwargs...)
        -> (E1, mps, energy_history)

First-excited-state search by adding a penalty `weight·|ψ0⟩⟨ψ0|` to the
effective two-site Hamiltonian at every bond (standard DMRG deflation trick),
where `ψ0` is a fixed, already-converged ground-state MPS. `weight` should
exceed the true spectral gap (defaults to `10·|E0|` heuristically via the
ground MPS's own energy if not given).
"""
function dmrg_excited_state(
    mpo::Vector{<:Array{Float64,4}}, ground_mps::Vector{<:Array{Float64,3}};
    maxdim::Int=32, n_sweeps::Int=10, cutoff::Float64=1e-14,
    weight::Union{Nothing,Float64}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    mps0::Union{Nothing,Vector{<:Array{Float64,3}}}=nothing,
    krylovdim::Int=12, krylov_tol::Float64=1e-10,
    verbose::Bool=false,
)
    N = length(mpo)
    dims = [size(W, 2) for W in mpo]
    ψ0 = [Array{Float64,3}(A) for A in ground_mps]
    E0 = dmrg_energy(ψ0, mpo)
    w = weight === nothing ? 10.0 * (abs(E0) + 1.0) : weight

    mps = mps0 === nothing ? random_mps_hetero(dims, maxdim; rng=rng) :
          [Array{Float64,3}(A) for A in mps0]
    right_canonicalize_mps!(mps)

    Lenv, Renv = mpo_environments(mps, mpo)
    # Overlap (identity-MPO) environments between the trial state and ψ0.
    OLenv, ORenv = _overlap_environments(mps, ψ0)

    energy_history = Float64[]
    E = NaN
    for sweep in 1:n_sweeps
        for j in 1:(N - 1)
            psi0 = merge_two_site(mps[j], mps[j + 1])
            proj = _overlap_two_site(OLenv[j], ψ0[j], ψ0[j + 1], ORenv[j + 3])
            u = proj ./ max(norm(proj), 1e-300)
            f = x -> begin
                p = reshape(x, size(psi0))
                hv = apply_Heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], p)
                pen = w * dot(vec(u), vec(p)) .* u
                vec(hv .+ pen)
            end
            vals, vecs, info = eigsolve(f, vec(psi0), 1, :SR; ishermitian=true,
                                        tol=krylov_tol, krylovdim=krylovdim, maxiter=100)
            psi_opt = reshape(vecs[1], size(psi0))
            E_raw = vals[1]
            E = real(dot(vec(psi_opt), vec(apply_Heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], psi_opt)))) / dot(vec(psi_opt), vec(psi_opt))
            A1, A2, _ = split_two_site(psi_opt, maxdim, :right; cutoff=cutoff)
            mps[j], mps[j + 1] = A1, A2
            Lenv[j + 1] = _update_Lenv(Lenv[j], mps[j], mpo[j])
            OLenv[j + 1] = _update_overlap_L(OLenv[j], mps[j], ψ0[j])
            push!(energy_history, E)
            verbose && println("  sweep $sweep  bond $j→$(j+1) (L→R)  E = $E  χ = $(size(A2,1))")
        end
        for j in (N - 1):-1:1
            psi0 = merge_two_site(mps[j], mps[j + 1])
            proj = _overlap_two_site(OLenv[j], ψ0[j], ψ0[j + 1], ORenv[j + 3])
            u = proj ./ max(norm(proj), 1e-300)
            f = x -> begin
                p = reshape(x, size(psi0))
                hv = apply_Heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], p)
                pen = w * dot(vec(u), vec(p)) .* u
                vec(hv .+ pen)
            end
            vals, vecs, info = eigsolve(f, vec(psi0), 1, :SR; ishermitian=true,
                                        tol=krylov_tol, krylovdim=krylovdim, maxiter=100)
            psi_opt = reshape(vecs[1], size(psi0))
            E = real(dot(vec(psi_opt), vec(apply_Heff_twosite(Lenv[j], mpo[j], mpo[j + 1], Renv[j + 3], psi_opt)))) / dot(vec(psi_opt), vec(psi_opt))
            A1, A2, _ = split_two_site(psi_opt, maxdim, :left; cutoff=cutoff)
            mps[j], mps[j + 1] = A1, A2
            Renv[j + 2] = _update_Renv(Renv[j + 3], mps[j + 1], mpo[j + 1])
            ORenv[j + 2] = _update_overlap_R(ORenv[j + 3], mps[j + 1], ψ0[j + 1])
            push!(energy_history, E)
            verbose && println("  sweep $sweep  bond $(j+1)→$j (R→L)  E = $E  χ = $(size(A1,3))")
        end
    end
    return E, mps, energy_history
end

function _update_overlap_L(OL::Matrix{Float64}, A::Array{Float64,3}, A0::Array{Float64,3})
    @tensor ONew[bp, b] := OL[ap, a] * A[ap, s, bp] * A0[a, s, b]
    return ONew
end

function _update_overlap_R(OR::Matrix{Float64}, A::Array{Float64,3}, A0::Array{Float64,3})
    @tensor ONew[ap, a] := OR[bp, b] * A[ap, s, bp] * A0[a, s, b]
    return ONew
end

function _overlap_environments(mps::Vector{<:Array{Float64,3}}, ψ0::Vector{<:Array{Float64,3}})
    N = length(mps)
    OL = Vector{Matrix{Float64}}(undef, N + 1)
    OR = Vector{Matrix{Float64}}(undef, N + 2)
    OL[1] = ones(Float64, 1, 1)
    OR[N + 2] = ones(Float64, 1, 1)
    @inbounds for j in 1:N
        OL[j + 1] = _update_overlap_L(OL[j], Array{Float64,3}(mps[j]), ψ0[j])
    end
    @inbounds for j in N:-1:1
        OR[j + 1] = _update_overlap_R(OR[j + 2], Array{Float64,3}(mps[j]), ψ0[j])
    end
    return OL, OR
end

"""Two-site tensor of `ψ0` expressed in the trial state's current left/right
bond bases via overlap environments — the local "target" for deflation."""
function _overlap_two_site(OL::Matrix{Float64}, A01::Array{Float64,3}, A02::Array{Float64,3}, OR::Matrix{Float64})
    @tensor proj[a, s1, s2, b] := OL[a, a0] * A01[a0, s1, m] * A02[m, s2, b0] * OR[b, b0]
    return proj
end

export random_mps_hetero, mps_inner_product, apply_mpo_to_mps
export dmrg_residual, dmrg_energy, mpo_environments
export apply_Heff_twosite, merge_two_site, split_two_site
export dmrg_ground_state, dmrg_excited_state
