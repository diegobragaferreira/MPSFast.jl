# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Binary Tree Tensor Network (BTT) Born Machine  ─  MPSFast.jl
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Heap layout (1-based):
#    root = 1,  left(i) = 2i,  right(i) = 2i+1,  parent(i) = i >> 1
#    internal nodes: 1 … N_pad-1
#    leaves:         N_pad … 2*N_pad-1   (leaf index j = h - N_pad + 1)
#
#  Tensor shapes:
#    leaf j   (d > 1):  (d_j, χ_j)          physical × upward bond
#    leaf j   (d = 1):  (1, 1)               trivial padding (fixed = [[1.0]])
#    internal i ≠ 1:   (χ_l, χ_r, χ_u)     left-child bond × right-child bond × upward bond
#    root (i = 1):      (χ_l, χ_r, 1)        upward bond = 1 (no parent)
#
#  Key convention matching existing code:
#    reshape(B, χ_l*χ_r, χ_u)  ← Julia column-major; index k = r_l + χ_l*(r_r-1)
#    kron(M_r, M_l)             ← matches above reshape (right-major ⊗ left-minor)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── Type ────────────────────────────────────────────────────────────────────

"""
    BinaryTTN{T<:Real}

Born-machine Binary Tree Tensor Network.

# Fields
- `leaves`:    `Vector` of `N_pad` leaf tensors; `leaves[j]` has shape `(d_j, χ_j)`.
- `internals`: `Vector` of `N_pad-1` internal tensors; `internals[i]` has shape
               `(χ_l, χ_r, χ_u)`; root = `internals[1]` with `χ_u = 1`.
- `n_sites`:   number of physical sites (un-padded).
- `N_pad`:     next power of 2 ≥ `n_sites`.
- `d_vec`:     local dimensions per leaf (`d_vec[j] = 1` for padding leaves).
- `D_max`:     maximum bond dimension.
"""
struct BinaryTTN{T<:Real}
    leaves   :: Vector{Matrix{T}}      # leaves[j]:    (d_j, χ_j)
    internals:: Vector{Array{T,3}}     # internals[i]: (χ_l, χ_r, χ_u)
    n_sites  :: Int
    N_pad    :: Int
    d_vec    :: Vector{Int}
    D_max    :: Int
end

# ─── Topology helpers ────────────────────────────────────────────────────────

@inline _ttn_left(i)           = 2i
@inline _ttn_right(i)          = 2i + 1
@inline _ttn_parent(i)         = i >> 1
@inline _ttn_is_leaf(h, Np)    = h >= Np
@inline _ttn_leaf_idx(h, Np)   = h - Np + 1          # 1-based index into leaves[]
@inline _ttn_heap_of_leaf(j,Np)= Np + j - 1

# ─── Initialisation ──────────────────────────────────────────────────────────

"""
    init_ttn(n_sites, d, D_max; T=Float32, rng=Random.default_rng()) -> BinaryTTN{T}

Randomly initialise a `BinaryTTN` Born machine.  `n_sites` physical sites each
with local dimension `d`; bond dimensions capped at `D_max`.  Sites beyond the
next power-of-2 padding are trivial (fixed, never updated).
"""
function init_ttn(n_sites::Int, d::Int, D_max::Int;
                  T::Type{<:Real}  = Float32,
                  rng              = Random.default_rng())
    N_pad = nextpow(2, max(n_sites, 2))
    d_vec = [j <= n_sites ? d : 1 for j in 1:N_pad]
    return _init_ttn_from_dvec(n_sites, d_vec, N_pad, D_max; T = T, rng = rng)
end

"""
    init_ttn_classification(M_path, d_path, n_classes, D_max; kwargs...) -> BinaryTTN

Joint TTN over `M_path` encoded path sites plus one label leaf (`n_classes` outcomes).
"""
function init_ttn_classification(M_path::Int, d_path::Int, n_classes::Int, D_max::Int;
                                 T::Type{<:Real}  = Float32,
                                 rng              = Random.default_rng())
    n_sites = M_path + 1
    N_pad   = nextpow(2, max(n_sites, 2))
    d_vec   = [j <= M_path ? d_path : (j == n_sites ? n_classes : 1) for j in 1:N_pad]
    return _init_ttn_from_dvec(n_sites, d_vec, N_pad, D_max; T = T, rng = rng)
end

function _init_ttn_from_dvec(n_sites::Int, d_vec::Vector{Int}, N_pad::Int, D_max::Int;
                             T::Type{<:Real}  = Float32,
                             rng              = Random.default_rng())

    # Upward bond dimension at each heap node (saturates at D_max quickly).
    χ_up = zeros(Int, 2N_pad)
    for j in 1:N_pad
        χ_up[N_pad + j - 1] = min(D_max, d_vec[j])
    end
    for h in (N_pad-1):-1:2          # skip root (h=1 has no upward bond)
        χ_up[h] = min(D_max, χ_up[2h] * χ_up[2h+1])
    end
    χ_up[1] = 1                       # root: virtual upward bond = 1

    # ── Leaves ──
    leaves = Vector{Matrix{T}}(undef, N_pad)
    for j in 1:N_pad
        h = N_pad + j - 1
        dj = d_vec[j]; χ = χ_up[h]
        if dj == 1
            leaves[j] = ones(T, 1, 1)
        else
            A = randn(rng, T, dj, χ)
            F = qr(A); Q = Matrix(F.Q)
            leaves[j] = T.(Q[:, 1:min(dj, χ)])
        end
    end

    # ── Internals ──
    internals = Vector{Array{T,3}}(undef, N_pad - 1)
    for h in 1:(N_pad-1)
        hl = 2h; hr = 2h + 1
        χ_l = χ_up[hl]; χ_r = χ_up[hr]
        χ_u = (h == 1) ? 1 : χ_up[h]
        B   = randn(rng, T, χ_l, χ_r, χ_u)
        if h == 1
            B ./= T(norm(B))
        else
            Bm = reshape(B, χ_l * χ_r, χ_u)
            F  = qr(Bm); Q = Matrix(F.Q)
            B  = reshape(T.(Q[:, 1:min(χ_l*χ_r, χ_u)]), χ_l, χ_r, χ_u)
        end
        internals[h] = B
    end

    return BinaryTTN{T}(leaves, internals, n_sites, N_pad, d_vec, D_max)
end

# ─── Canonical form (root-canonical: all nodes upward-isometric) ─────────────

"""
    root_canonicalize_ttn!(ttn)

Make the `BinaryTTN` root-canonical: sweep leaves then internals bottom-up via
QR, absorbing the R factor into the parent.  After this call `Z = ‖root‖²_F`
and all non-root upward norm matrices equal the identity.
"""
function root_canonicalize_ttn!(ttn::BinaryTTN{T}) where T
    N_pad = ttn.N_pad

    # ── Leaves ──────────────────────────────────────────────────────────────
    for j in 1:N_pad
        ttn.d_vec[j] == 1 && continue        # skip trivial padding
        h = N_pad + j - 1
        A = ttn.leaves[j]                     # (d, χ)
        d, χ = size(A)
        k    = min(d, χ)
        F    = qr(A)
        Q    = T.(Matrix(F.Q)[:, 1:k])
        R    = T.(Matrix(F.R)[1:k, :])
        ttn.leaves[j] = Q

        # Absorb R into the bond connecting to parent.
        hp = _ttn_parent(h)
        hp == 0 && continue
        _ttn_absorb_R_into_child_bond!(ttn, hp, h, R, k)
    end

    # ── Internals (bottom-up, skipping root) ────────────────────────────────
    for h in (N_pad-1):-1:2
        B         = ttn.internals[h]          # (χ_l, χ_r, χ_u)
        χ_l, χ_r, χ_u = size(B)
        k         = min(χ_l * χ_r, χ_u)
        Bm        = reshape(B, χ_l * χ_r, χ_u)
        F         = qr(Bm)
        Q         = T.(Matrix(F.Q)[:, 1:k])
        R         = T.(Matrix(F.R)[1:k, :])
        ttn.internals[h] = reshape(Q, χ_l, χ_r, k)

        hp = _ttn_parent(h)
        _ttn_absorb_R_into_child_bond!(ttn, hp, h, R, k)
    end
    # Root is left untouched and carries all the gauge freedom.
end

# Absorb factor R (k × χ_old) into the bond from parent `hp` toward child `h`.
function _ttn_absorb_R_into_child_bond!(ttn::BinaryTTN{T},
                                         hp::Int, h::Int,
                                         R::Matrix{T}, k::Int) where T
    Bp       = ttn.internals[hp]             # (χ_l, χ_r, χ_u)
    χ_l, χ_r, χ_u = size(Bp)

    if _ttn_left(hp) == h
        # h is left child of hp: R is (k × χ_l) from the child QR.
        Bp_new = reshape(R * reshape(Bp, χ_l, χ_r * χ_u), k, χ_r, χ_u)
        ttn.internals[hp] = T.(Bp_new)
    else
        # h is right child: second index of Bp.
        Bt     = permutedims(Bp, (2, 1, 3))    # (χ_r, χ_l, χ_u)
        Bt_new = reshape(R * reshape(Bt, χ_r, χ_l * χ_u), k, χ_l, χ_u)
        ttn.internals[hp] = T.(permutedims(Bt_new, (2, 1, 3)))
    end
end

# ─── Single-sample amplitude ─────────────────────────────────────────────────

"""
    ttn_amplitude(ttn, x) -> Float64

Compute ψ(x) = contraction of the TTN for physical indices `x` (1-based,
length `n_sites`).  Cost O(n_sites · χ³).
"""
function ttn_amplitude(ttn::BinaryTTN{T}, x::AbstractVector{Int}) where T
    N_pad = ttn.N_pad
    u = Vector{Vector{Float64}}(undef, 2N_pad)

    # Leaves
    for j in 1:N_pad
        h = N_pad + j - 1
        if ttn.d_vec[j] == 1
            u[h] = [1.0]
        else
            u[h] = Float64.(ttn.leaves[j][x[j], :])
        end
    end

    # Internals (bottom-up)
    for h in (N_pad-1):-1:1
        hl = 2h; hr = 2h + 1
        B  = Float64.(ttn.internals[h])      # (χ_l, χ_r, χ_u)
        χ_l, χ_r, χ_u = size(B)
        ul = u[hl]; ur = u[hr]
        # u[h][r_u] = Σ_{r_l,r_r} B[r_l,r_r,r_u] * ul[r_l] * ur[r_r]
        #           = B_mat' * (ul ⊗ ur)   where B_mat = reshape(B, χ_l*χ_r, χ_u)
        #           and (ul ⊗ ur)[r_l + χ_l*(r_r-1)] = ul[r_l]*ur[r_r]  (col-major kron)
        outer = vec(ul * ur')               # (χ_l*χ_r,)  col-major = ul ⊗ ur
        u[h]  = reshape(B, χ_l * χ_r, χ_u)' * outer
    end
    return u[1][1]
end

# ─── Batched upward pass ─────────────────────────────────────────────────────

# U[h] is (N_d, χ_h): upward vectors for all training samples at heap node h.
function _ttn_upward!(U::Vector{Matrix{Float64}},
                      ttn::BinaryTTN{T},
                      xi_data::AbstractMatrix{Int}) where T
    N_d   = size(xi_data, 1)
    N_pad = ttn.N_pad

    # Leaves
    for j in 1:N_pad
        h = N_pad + j - 1
        if ttn.d_vec[j] == 1
            U[h] = ones(Float64, N_d, 1)
        else
            A  = Float64.(ttn.leaves[j])         # (d, χ)
            χ  = size(A, 2)
            Uh = Matrix{Float64}(undef, N_d, χ)
            @inbounds for i in 1:N_d
                σ = xi_data[i, j]
                @views Uh[i, :] .= A[σ, :]
            end
            U[h] = Uh
        end
    end

    # Internals (bottom-up)
    for h in (N_pad-1):-1:1
        hl = 2h; hr = 2h + 1
        Ul  = U[hl]; Ur = U[hr]
        χ_l = size(Ul, 2); χ_r = size(Ur, 2)
        B   = Float64.(ttn.internals[h])         # (χ_l, χ_r, χ_u)
        χ_u = size(B, 3)
        # KR[i, r_l + χ_l*(r_r-1)] = Ul[i,r_l] * Ur[i,r_r]
        KR  = reshape(reshape(Ul, N_d, χ_l, 1) .* reshape(Ur, N_d, 1, χ_r), N_d, χ_l * χ_r)
        U[h]= KR * reshape(B, χ_l * χ_r, χ_u)   # (N_d, χ_u)
    end
end

# ─── Batched downward pass ───────────────────────────────────────────────────

# E_down[h] is (N_d, χ_u_h): downward environments for all samples at node h.
function _ttn_downward!(E_down::Vector{Matrix{Float64}},
                         ttn::BinaryTTN{T},
                         U::Vector{Matrix{Float64}}) where T
    N_d   = size(U[ttn.N_pad], 1)
    N_pad = ttn.N_pad
    E_down[1] = ones(Float64, N_d, 1)           # root: scalar env = 1

    for h in 1:(N_pad-1)
        hl = 2h; hr = 2h + 1
        B  = Float64.(ttn.internals[h])          # (χ_l, χ_r, χ_u)
        χ_l, χ_r, χ_u = size(B)
        Eh = E_down[h]                           # (N_d, χ_u)

        # ── Downward env for left child ──────────────────────────────────────
        # E_down[hl][i,r_l] = Σ_{r_r,r_u} B[r_l,r_r,r_u] * Eh[i,r_u] * Ur[i,r_r]
        # Step 1: contract B with Ur over r_r → shape (N_d, χ_l*χ_u) [col-major: r_l fastest]
        # permutedims(B,(1,3,2)) → (χ_l, χ_u, χ_r); reshape → (χ_l*χ_u, χ_r)
        B_lr  = reshape(permutedims(B, (1,3,2)), χ_l * χ_u, χ_r)
        tmp_l = U[hr] * B_lr'                    # (N_d, χ_l*χ_u)
        tmp_l3= reshape(tmp_l, N_d, χ_l, χ_u)
        # Step 2: contract with Eh over χ_u
        E_down[hl] = dropdims(
            sum(tmp_l3 .* reshape(Eh, N_d, 1, χ_u), dims=3), dims=3)  # (N_d, χ_l)

        # ── Downward env for right child ─────────────────────────────────────
        # E_down[hr][i,r_r] = Σ_{r_l,r_u} B[r_l,r_r,r_u] * Eh[i,r_u] * Ul[i,r_l]
        # permutedims(B,(2,3,1)) → (χ_r, χ_u, χ_l); reshape → (χ_r*χ_u, χ_l)
        B_rr  = reshape(permutedims(B, (2,3,1)), χ_r * χ_u, χ_l)
        tmp_r = U[hl] * B_rr'                    # (N_d, χ_r*χ_u)
        tmp_r3= reshape(tmp_r, N_d, χ_r, χ_u)
        E_down[hr] = dropdims(
            sum(tmp_r3 .* reshape(Eh, N_d, 1, χ_u), dims=3), dims=3)  # (N_d, χ_r)
    end
end

# ─── Shared contraction helpers (O(χ³), no kron expansion) ───────────────────
#
# Three dispatch variants for propagating the downward norm matrix one level
# toward a child.  All replace the original O(χ⁶) kron expansion.
#
#   _I       M_sib = identity (unsampled, post-canonical)  → 1 DGEMM,   O(χ³)
#   _rank1   M_sib = u⊗u (fixed upward vector)            → 2 GEMVs,   O(χ³)
#   default  M_sib arbitrary (gradient computation)        → loop,      O(χ⁴)

function _mdown_step_left_I(B::Array{Float64,3}, Mdn::Matrix{Float64})
    χ_l, χ_r, χ_u = size(B)
    # Σ_{r_r,v,v'} B[r_l,r_r,v]*Mdn[v,v']*B[r_l',r_r,v']
    # = reshape(Bmat*Mdn, χ_l, χ_r*χ_u) * reshape(B, χ_l, χ_r*χ_u)'
    Tc = reshape(reshape(B, χ_l * χ_r, χ_u) * Mdn, χ_l, χ_r * χ_u)
    Bm = reshape(B, χ_l, χ_r * χ_u)
    return Tc * Bm'
end

function _mdown_step_right_I(B::Array{Float64,3}, Mdn::Matrix{Float64})
    χ_l, χ_r, χ_u = size(B)
    # Σ_{r_l,v,v'} B[r_l,r_r,v]*Mdn[v,v']*B[r_l,r_r',v']
    # = Tmat * Bmat'  where Tmat[r_r,(r_l,v)] after permutedims
    Tc3d = reshape(reshape(B, χ_l * χ_r, χ_u) * Mdn, χ_l, χ_r, χ_u)
    Tmat = reshape(permutedims(Tc3d, (2,1,3)), χ_r, χ_l * χ_u)
    Bmat = reshape(permutedims(B,    (2,1,3)), χ_r, χ_l * χ_u)
    return Tmat * Bmat'
end

function _mdown_step_left_rank1(B::Array{Float64,3}, u::Vector{Float64}, Mdn::Matrix{Float64})
    χ_l, χ_r, χ_u = size(B)
    Tc3 = reshape(reshape(B, χ_l * χ_r, χ_u) * Mdn, χ_l, χ_r, χ_u)
    cT  = reshape(reshape(permutedims(Tc3, (1,3,2)), χ_l * χ_u, χ_r) * u, χ_l, χ_u)
    cB  = reshape(reshape(permutedims(B,   (1,3,2)), χ_l * χ_u, χ_r) * u, χ_l, χ_u)
    return cT * cB'
end

function _mdown_step_right_rank1(B::Array{Float64,3}, u::Vector{Float64}, Mdn::Matrix{Float64})
    χ_l, χ_r, χ_u = size(B)
    Tc3 = reshape(reshape(B, χ_l * χ_r, χ_u) * Mdn, χ_l, χ_r, χ_u)
    cT  = reshape(reshape(permutedims(Tc3, (2,3,1)), χ_r * χ_u, χ_l) * u, χ_r, χ_u)
    cB  = reshape(reshape(permutedims(B,   (2,3,1)), χ_r * χ_u, χ_l) * u, χ_r, χ_u)
    return cT * cB'
end

function _mdown_step_left(B::Array{Float64,3}, M_sib_r::Matrix{Float64}, Mdn::Matrix{Float64})
    χ_l, χ_r, χ_u = size(B)
    Tc  = reshape(reshape(B, χ_l * χ_r, χ_u) * Mdn, χ_l, χ_r, χ_u)
    res = zeros(Float64, χ_l, χ_l)
    tmp = Matrix{Float64}(undef, χ_l, χ_r)
    @inbounds for v in 1:χ_u
        mul!(tmp, view(Tc,:,:,v), M_sib_r)
        mul!(res, tmp, view(B,:,:,v)', 1.0, 1.0)
    end
    return res
end

function _mdown_step_right(B::Array{Float64,3}, M_sib_l::Matrix{Float64}, Mdn::Matrix{Float64})
    χ_l, χ_r, χ_u = size(B)
    Tc  = reshape(reshape(B, χ_l * χ_r, χ_u) * Mdn, χ_l, χ_r, χ_u)
    res = zeros(Float64, χ_r, χ_r)
    tmp = Matrix{Float64}(undef, χ_l, χ_r)
    @inbounds for v in 1:χ_u
        mul!(tmp, M_sib_l, view(B,:,:,v))
        mul!(res, view(Tc,:,:,v)', tmp, 1.0, 1.0)
    end
    return res
end

# ─── Upward norm matrices (for Z and ∂Z) ─────────────────────────────────────

# M_up[h] = Σ_x u[h](x) u[h](x)' = (χ_h × χ_h) positive semidefinite matrix.
# After root_canonicalize_ttn!: M_up[h] = I for all h ≠ root.
#
# Uses efficient O(χ³) factored contraction instead of O(χ⁵) kron expansion.
function _ttn_norm_up(ttn::BinaryTTN{T}) where T
    N_pad = ttn.N_pad
    M     = Vector{Matrix{Float64}}(undef, 2N_pad)

    # Leaves: M_up[h] = A'A
    for j in 1:N_pad
        h = N_pad + j - 1
        A = Float64.(ttn.leaves[j])
        M[h] = A' * A
    end

    # Internals (bottom-up).
    # M_up[h][r_u,r_u'] = Σ_{r_l,r_r,r_l',r_r'} B[r_l,r_r,r_u]*M_l[r_l,r_l']*M_r[r_r,r_r']*B[r_l',r_r',r_u']
    # Factored:
    #   Step 1: T  = reshape(M_l * reshape(B, χ_l, χ_r*χ_u), χ_l, χ_r, χ_u)
    #   Step 2: S  = for each v, S[:,:,v] = T[:,:,v] * M_r   (via permutedims trick)
    #   Step 3: M_up[h] = reshape(S, χ_l*χ_r, χ_u)' * reshape(B, χ_l*χ_r, χ_u)
    for h in (N_pad-1):-1:1
        hl = 2h; hr = 2h + 1
        B    = Float64.(ttn.internals[h])
        χ_l, χ_r, χ_u = size(B)
        Ml   = M[hl]; Mr = M[hr]
        Tc   = reshape(Ml * reshape(B, χ_l, χ_r * χ_u), χ_l, χ_r, χ_u)
        # Contract Tc with Mr on the r_r dimension for each χ_u slice.
        Tp   = reshape(permutedims(Tc, (1,3,2)), χ_l * χ_u, χ_r)
        S    = permutedims(reshape(Tp * Mr, χ_l, χ_u, χ_r), (1,3,2))  # (χ_l, χ_r, χ_u)
        M[h] = reshape(S, χ_l * χ_r, χ_u)' * reshape(B, χ_l * χ_r, χ_u)
    end
    return M
end

function _ttn_logZ(ttn::BinaryTTN{T}, M_up::Vector{Matrix{Float64}}) where T
    Br        = Float64.(ttn.internals[1])       # (χ_l, χ_r, 1)
    bmat      = Br[:, :, 1]                      # (χ_l, χ_r)
    Ml        = M_up[2]; Mr = M_up[3]            # left child=2, right child=3
    # Z = Tr[bmat' * Ml * bmat * Mr]  = sum((Ml*bmat) .* (bmat*Mr))  [M_up symmetric]
    Z = sum((Ml * bmat) .* (bmat * Mr))
    return log(max(Z, 1e-30))
end

# ─── NLL estimate ────────────────────────────────────────────────────────────

"""
    ttn_nll(ttn, xi_data; n_samples=2000) -> Float64

Estimate negative log-likelihood on `n_samples` rows of `xi_data`.
"""
function ttn_nll(ttn::BinaryTTN{T}, xi_data::AbstractMatrix{Int};
                 n_samples::Int = 2000) where T
    N_d   = size(xi_data, 1)
    idx   = randperm(N_d)[1:min(n_samples, N_d)]
    M_up  = _ttn_norm_up(ttn)
    logZ  = _ttn_logZ(ttn, M_up)
    acc   = Threads.Atomic{Float64}(0.0)
    Threads.@threads for k in 1:length(idx)
        ψ = ttn_amplitude(ttn, xi_data[idx[k], :])
        Threads.atomic_add!(acc, logZ - 2.0 * log(max(abs(ψ), 1e-30)))
    end
    return acc[] / length(idx)
end

# ─── Downward norm matrix (for ∂Z/∂T_n without canonical form) ───────────────

# Walk from root down to node h_target, computing M_down[h_target].
# Uses M_up matrices for sibling subtrees.
function _ttn_mdown(ttn::BinaryTTN{T}, h_target::Int,
                    M_up::Vector{Matrix{Float64}}) where T
    N_pad = ttn.N_pad

    # Build path from root to h_target
    path = Int[]
    h = h_target
    while h > 1
        pushfirst!(path, h)
        h = _ttn_parent(h)
    end
    pushfirst!(path, 1)   # root

    M_dn = ones(Float64, 1, 1)   # M_down at root = 1×1 identity

    for k in 1:(length(path)-1)
        hp = path[k]
        hc = path[k+1]
        B  = Float64.(ttn.internals[hp])
        hl = _ttn_left(hp); hr = _ttn_right(hp)
        if hc == hl
            M_dn = _mdown_step_left(B, M_up[hr], M_dn)
        else
            M_dn = _mdown_step_right(B, M_up[hl], M_dn)
        end
    end
    return M_dn
end

# ─── Per-node gradient ───────────────────────────────────────────────────────

# Returns gradient array with same shape as the tensor at heap node h.
# Data term uses batched BLAS; norm term uses M_up / M_down matrices.
function _ttn_grad_node(ttn      :: BinaryTTN{T},
                         xi_data  :: AbstractMatrix{Int},
                         h        :: Int,
                         U        :: Vector{Matrix{Float64}},
                         E_down   :: Vector{Matrix{Float64}},
                         psi      :: Vector{Float64},
                         M_up     :: Vector{Matrix{Float64}}) where T
    N_d   = size(xi_data, 1)
    N_pad = ttn.N_pad
    Z     = exp(_ttn_logZ(ttn, M_up))

    # Inverse amplitudes (clamp to avoid NaN)
    w = @. ifelse(abs(psi) < 1e-30, 0.0, 1.0 / psi)   # (N_d,)

    if _ttn_is_leaf(h, N_pad)
        j  = _ttn_leaf_idx(h, N_pad)
        A  = Float64.(ttn.leaves[j])                     # (d, χ_u)
        d, χ_u = size(A)
        Ed = E_down[h]                                   # (N_d, χ_u)

        # Data term: ddata[σ, r_u] = -(2/N_d) Σ_{i:x_i=σ} w_i * Ed[i, r_u]
        ddata = zeros(Float64, d, χ_u)
        xj    = xi_data[:, j]
        WEd   = w .* Ed                                  # (N_d, χ_u)
        @inbounds for i in 1:N_d
            σ = xj[i]
            @views ddata[σ, :] .+= WEd[i, :]
        end
        ddata .*= -2.0 / N_d

        # Norm term: (2/Z) * A * M_down[h]
        M_dn  = _ttn_mdown(ttn, h, M_up)                # (χ_u, χ_u)
        dnorm = (2.0 / Z) .* (A * M_dn)

        return ddata .+ dnorm

    else
        B     = Float64.(ttn.internals[h])               # (χ_l, χ_r, χ_u)
        χ_l, χ_r, χ_u = size(B)
        hl    = _ttn_left(h); hr = _ttn_right(h)
        Ul    = U[hl]; Ur = U[hr]                        # (N_d, χ_l), (N_d, χ_r)
        Ed    = E_down[h]                                # (N_d, χ_u)

        # Data term (BLAS):
        # ddata_mat[r_l+χ_l*(r_r-1), r_u] = -(2/N_d) Σ_i w_i * KR[i,r_l*r_r] * Ed[i,r_u]
        KR       = reshape(reshape(Ul, N_d, χ_l, 1) .* reshape(Ur, N_d, 1, χ_r),
                           N_d, χ_l * χ_r)              # (N_d, χ_l*χ_r)
        wKR      = w .* KR                              # (N_d, χ_l*χ_r)  [broadcast w col]
        ddata_m  = (-2.0 / N_d) .* (wKR' * Ed)         # (χ_l*χ_r, χ_u)
        ddata    = reshape(ddata_m, χ_l, χ_r, χ_u)

        # Norm term (O(χ³), no kron):
        # dnorm[r_l,r_r,r_u] = (2/Z)*Σ_{r_l',r_r',r_u'} M_l[r_l,r_l']*M_r[r_r,r_r']*B[r_l',r_r',r_u']*M_dn[r_u',r_u]
        # Factor: T = reshape(B_mat*M_dn, χ_l,χ_r,χ_u),  then contract M_r on r_r, then M_l on r_l.
        M_dn  = h == 1 ? ones(Float64, 1, 1) : _ttn_mdown(ttn, h, M_up)
        Bm    = reshape(B, χ_l * χ_r, χ_u)
        Tp    = reshape(Bm * M_dn, χ_l, χ_r, χ_u)
        Tp2   = reshape(permutedims(Tp, (1,3,2)), χ_l * χ_u, χ_r)
        S     = permutedims(reshape(Tp2 * M_up[hr], χ_l, χ_u, χ_r), (1,3,2))   # (χ_l,χ_r,χ_u)
        dnorm = reshape((2.0 / Z) .* (M_up[hl] * reshape(S, χ_l, χ_r * χ_u)), χ_l, χ_r, χ_u)

        return ddata .+ dnorm
    end
end

# ─── Adam step (TTN-specific; reuses AdamSlot/AdamDict from training.jl) ─────

function _ttn_adam_leaf!(ttn :: BinaryTTN{T},
                          adam :: Dict{Int, Any},
                          h    :: Int,
                          grad :: Matrix{Float64},
                          η_t  :: Float64) where T
    j    = _ttn_leaf_idx(h, ttn.N_pad)
    d, χ = size(ttn.leaves[j])
    if !haskey(adam, h)
        adam[h] = (m = zeros(d, χ), v = zeros(d, χ), t = Ref(0))
    end
    s = adam[h]; s.t[] += 1
    β1 = 0.9; β2 = 0.999; ε = 1e-8; t = s.t[]
    @. s.m = β1 * s.m + (1 - β1) * grad
    @. s.v = β2 * s.v + (1 - β2) * grad * grad
    bc1 = 1 - β1^t; bc2 = 1 - β2^t
    step = @. η_t * (s.m / bc1) / (sqrt(s.v / bc2) + ε)
    ttn.leaves[j] .-= T.(step)
end

function _ttn_adam_internal!(ttn :: BinaryTTN{T},
                               adam :: Dict{Int, Any},
                               h    :: Int,
                               grad :: Array{Float64,3},
                               η_t  :: Float64) where T
    χ_l, χ_r, χ_u = size(ttn.internals[h])
    if !haskey(adam, h)
        adam[h] = (m = zeros(χ_l, χ_r, χ_u), v = zeros(χ_l, χ_r, χ_u), t = Ref(0))
    end
    s = adam[h]; s.t[] += 1
    β1 = 0.9; β2 = 0.999; ε = 1e-8; t = s.t[]
    @. s.m = β1 * s.m + (1 - β1) * grad
    @. s.v = β2 * s.v + (1 - β2) * grad * grad
    bc1 = 1 - β1^t; bc2 = 1 - β2^t
    step = @. η_t * (s.m / bc1) / (sqrt(s.v / bc2) + ε)
    ttn.internals[h] .-= T.(step)
end

# ─── Sweep order (zigzag) ────────────────────────────────────────────────────

"""Depth of internal heap node `h` from the root (`h = 1` → depth 0)."""
@inline _ttn_internal_depth(h::Int) = Int(floor(log2(h)))

"""Both children of internal `h` are physical leaves."""
function _ttn_is_leaf_parent(h::Int, N_pad::Int)
    hl = 2h; hr = 2h + 1
    return _ttn_is_leaf(hl, N_pad) && _ttn_is_leaf(hr, N_pad)
end

"""Internal nodes whose children are both leaves (e.g. `h = 4…7` when `N_pad = 8`)."""
function _ttn_leaf_parent_nodes(N_pad::Int)
    [h for h in 1:(N_pad - 1) if _ttn_is_leaf_parent(h, N_pad)]
end

"""
    _ttn_zigzag_internals(N_pad; bottom_up=true) -> Vector{Int}

Boustrophedon order over internal nodes: alternate L→R / R→L by depth from root.
"""
function _ttn_zigzag_internals(N_pad::Int; bottom_up::Bool = true)
    by_depth = Dict{Int, Vector{Int}}()
    for h in 1:(N_pad - 1)
        d = _ttn_internal_depth(h)
        push!(get!(by_depth, d, Int[]), h)
    end
    depths = sort(collect(keys(by_depth)))
    bottom_up && reverse!(depths)
    order = Int[]
    for (i, d) in enumerate(depths)
        hs = sort(by_depth[d])
        iseven(i) && reverse!(hs)
        append!(order, hs)
    end
    return order
end

"""Build a sweep schedule of `(:leaf|:internal|:leaf_pair, id)` tasks."""
function _ttn_sweep_tasks(
    N_pad::Int, leaf_fwd::Vector{Int};
    bottom_up::Bool = true,
    dmrg_pairs::Bool = true,
)
    tasks = Tuple{Symbol, Int}[]
    if dmrg_pairs
        leaf_parents = Set(_ttn_leaf_parent_nodes(N_pad))
        for h in _ttn_zigzag_internals(N_pad; bottom_up = bottom_up)
            if h in leaf_parents
                push!(tasks, (:leaf_pair, h))
            else
                push!(tasks, (:internal, h))
            end
        end
    else
        leaves = bottom_up ? leaf_fwd : reverse(leaf_fwd)
        for j in leaves
            push!(tasks, (:leaf, j))
        end
        for h in _ttn_zigzag_internals(N_pad; bottom_up = bottom_up)
            push!(tasks, (:internal, h))
        end
    end
    return tasks
end

# ─── Incremental environment refresh ─────────────────────────────────────────

function _ttn_upward_one!(U::Vector{Matrix{Float64}},
                          ttn::BinaryTTN{T},
                          xi_data::AbstractMatrix{Int},
                          h::Int) where T
    N_d   = size(xi_data, 1)
    N_pad = ttn.N_pad
    if _ttn_is_leaf(h, N_pad)
        j = _ttn_leaf_idx(h, N_pad)
        if ttn.d_vec[j] == 1
            U[h] = ones(Float64, N_d, 1)
        else
            A  = Float64.(ttn.leaves[j])
            χ  = size(A, 2)
            Uh = Matrix{Float64}(undef, N_d, χ)
            @inbounds for i in 1:N_d
                σ = xi_data[i, j]
                @views Uh[i, :] .= A[σ, :]
            end
            U[h] = Uh
        end
    else
        hl = 2h; hr = 2h + 1
        Ul = U[hl]; Ur = U[hr]
        χ_l = size(Ul, 2); χ_r = size(Ur, 2)
        B   = Float64.(ttn.internals[h])
        χ_u = size(B, 3)
        KR  = reshape(reshape(Ul, N_d, χ_l, 1) .* reshape(Ur, N_d, 1, χ_r), N_d, χ_l * χ_r)
        U[h] = KR * reshape(B, χ_l * χ_r, χ_u)
    end
    return nothing
end

function _ttn_upward_chain!(U::Vector{Matrix{Float64}},
                            ttn::BinaryTTN{T},
                            xi_data::AbstractMatrix{Int},
                            h::Int) where T
    while true
        _ttn_upward_one!(U, ttn, xi_data, h)
        h == 1 && break
        h = _ttn_parent(h)
    end
    return nothing
end

function _ttn_downward_children!(E_down::Vector{Matrix{Float64}},
                                 ttn::BinaryTTN{T},
                                 U::Vector{Matrix{Float64}},
                                 h::Int) where T
    hl = 2h; hr = 2h + 1
    B  = Float64.(ttn.internals[h])
    χ_l, χ_r, χ_u = size(B)
    Eh = E_down[h]
    B_lr = reshape(permutedims(B, (1, 3, 2)), χ_l * χ_u, χ_r)
    tmp_l = U[hr] * B_lr'
    tmp_l3 = reshape(tmp_l, size(U[hr], 1), χ_l, χ_u)
    E_down[hl] = dropdims(sum(tmp_l3 .* reshape(Eh, size(Eh, 1), 1, χ_u), dims = 3), dims = 3)
    B_rr = reshape(permutedims(B, (2, 3, 1)), χ_r * χ_u, χ_l)
    tmp_r = U[hl] * B_rr'
    tmp_r3 = reshape(tmp_r, size(U[hl], 1), χ_r, χ_u)
    E_down[hr] = dropdims(sum(tmp_r3 .* reshape(Eh, size(Eh, 1), 1, χ_u), dims = 3), dims = 3)
    return nothing
end

function _ttn_downward_subtree!(E_down::Vector{Matrix{Float64}},
                                ttn::BinaryTTN{T},
                                U::Vector{Matrix{Float64}},
                                h::Int) where T
    N_pad = ttn.N_pad
    h >= N_pad && return nothing
    _ttn_downward_children!(E_down, ttn, U, h)
    hl = 2h; hr = 2h + 1
    !_ttn_is_leaf(hl, N_pad) && _ttn_downward_subtree!(E_down, ttn, U, hl)
    !_ttn_is_leaf(hr, N_pad) && _ttn_downward_subtree!(E_down, ttn, U, hr)
    return nothing
end

function _ttn_norm_up_internal!(M::Vector{Matrix{Float64}},
                                ttn::BinaryTTN{T},
                                h::Int) where T
    hl = 2h; hr = 2h + 1
    B    = Float64.(ttn.internals[h])
    χ_l, χ_r, χ_u = size(B)
    Ml   = M[hl]; Mr = M[hr]
    Tc   = reshape(Ml * reshape(B, χ_l, χ_r * χ_u), χ_l, χ_r, χ_u)
    Tp   = reshape(permutedims(Tc, (1, 3, 2)), χ_l * χ_u, χ_r)
    S    = permutedims(reshape(Tp * Mr, χ_l, χ_u, χ_r), (1, 3, 2))
    M[h] = reshape(S, χ_l * χ_r, χ_u)' * reshape(B, χ_l * χ_r, χ_u)
    return nothing
end

function _ttn_norm_up_chain!(M::Vector{Matrix{Float64}},
                             ttn::BinaryTTN{T},
                             h::Int) where T
    N_pad = ttn.N_pad
    if _ttn_is_leaf(h, N_pad)
        j = _ttn_leaf_idx(h, N_pad)
        A = Float64.(ttn.leaves[j])
        M[h] = A' * A
    else
        _ttn_norm_up_internal!(M, ttn, h)
    end
    while h > 1
        h = _ttn_parent(h)
        _ttn_norm_up_internal!(M, ttn, h)
    end
    return nothing
end

"""Post-order upward refresh of `U` on the subtree rooted at heap node `h`."""
function _ttn_refresh_up_subtree!(U::Vector{Matrix{Float64}},
                                    ttn::BinaryTTN{T},
                                    xi_data::AbstractMatrix{Int},
                                    h::Int) where T
    N_pad = ttn.N_pad
    if _ttn_is_leaf(h, N_pad)
        _ttn_upward_one!(U, ttn, xi_data, h)
    else
        hl = 2h; hr = 2h + 1
        _ttn_refresh_up_subtree!(U, ttn, xi_data, hl)
        _ttn_refresh_up_subtree!(U, ttn, xi_data, hr)
        _ttn_upward_one!(U, ttn, xi_data, h)
    end
    return nothing
end

"""Post-order refresh of `M_up` on the subtree rooted at heap node `h`."""
function _ttn_norm_up_subtree!(M::Vector{Matrix{Float64}},
                               ttn::BinaryTTN{T},
                               h::Int) where T
    N_pad = ttn.N_pad
    if _ttn_is_leaf(h, N_pad)
        j = _ttn_leaf_idx(h, N_pad)
        A = Float64.(ttn.leaves[j])
        M[h] = A' * A
    else
        hl = 2h; hr = 2h + 1
        _ttn_norm_up_subtree!(M, ttn, hl)
        _ttn_norm_up_subtree!(M, ttn, hr)
        _ttn_norm_up_internal!(M, ttn, h)
    end
    return nothing
end

function _ttn_refresh_after_node!(
    U::Vector{Matrix{Float64}},
    E_down::Vector{Matrix{Float64}},
    M_up::Vector{Matrix{Float64}},
    ttn::BinaryTTN{T},
    xi_data::AbstractMatrix{Int},
    h::Int,
) where T
    N_pad = ttn.N_pad
    if _ttn_is_leaf(h, N_pad)
        _ttn_upward_chain!(U, ttn, xi_data, h)
        _ttn_norm_up_chain!(M_up, ttn, h)
    else
        _ttn_refresh_up_subtree!(U, ttn, xi_data, h)
        _ttn_norm_up_subtree!(M_up, ttn, h)
        _ttn_downward_subtree!(E_down, ttn, U, h)
        # ancestors above `h` may also need `M_up` if only subtree was refreshed
        hp = _ttn_parent(h)
        while hp >= 1
            _ttn_norm_up_internal!(M_up, ttn, hp)
            hp == 1 && break
            hp = _ttn_parent(hp)
        end
    end
    return U[1][:, 1]
end

# ─── DMRG-style leaf-pair update ─────────────────────────────────────────────

function update_ttn_leaf_pair!(
    ttn::BinaryTTN{T},
    adam::Dict{Int, Any},
    h::Int,
    xi_data::AbstractMatrix{Int},
    U::Vector{Matrix{Float64}},
    E_down::Vector{Matrix{Float64}},
    psi_all::Vector{Float64},
    M_up::Vector{Matrix{Float64}},
    η_t::Float64,
    D_max::Int,
    ε_cut::Real;
    sweep::Symbol = :none,
    bond_log = nothing,
    epoch::Int = 0,
    d_phys::Int = 0,
) where T
    N_pad = ttn.N_pad
    hl = 2h; hr = 2h + 1

    grad_B = _ttn_grad_node(ttn, xi_data, h,  U, E_down, psi_all, M_up)
    grad_l = _ttn_grad_node(ttn, xi_data, hl, U, E_down, psi_all, M_up)
    grad_r = _ttn_grad_node(ttn, xi_data, hr, U, E_down, psi_all, M_up)

    _clip_ttn_grad!(grad_B); _clip_ttn_grad!(grad_l); _clip_ttn_grad!(grad_r)

    _ttn_adam_internal!(ttn, adam, h, grad_B, η_t)
    _ttn_adam_leaf!(ttn, adam, hl, grad_l, η_t)
    _ttn_adam_leaf!(ttn, adam, hr, grad_r, η_t)

    # Truncated SVD on the sibling bond inside B_h: (χ_l·χ_r) × χ_u
    B = Float64.(ttn.internals[h])
    χ_l, χ_r, χ_u = size(B)
    d_phys == 0 && (d_phys = size(ttn.leaves[_ttn_leaf_idx(hl, N_pad)], 1))
    Bm = reshape(B, χ_l * χ_r, χ_u)
    keep_target = min(D_max, minimum(size(Bm)))
    keep_target == 0 && return nothing
    U_s, S, Vt = _truncated_svd(Bm, keep_target)
    keep = length(S)
    keep == 0 && return nothing
    if ε_cut > 0
        keep = max(1, min(sum(S .> Float64(ε_cut)), keep))
        U_s = U_s[:, 1:keep]; S = S[1:keep]; Vt = Vt[1:keep, :]
    end
    if sweep === :forward
        B_new = U_s[:, 1:keep] * Diagonal(S[1:keep]) * Vt[1:keep, :]
    else
        sqS = sqrt.(S[1:keep])
        B_new = (U_s[:, 1:keep] .* sqS') * (sqS .* Vt[1:keep, :])
    end
    ttn.internals[h] = T.(reshape(B_new, χ_l, χ_r, keep))

    if bond_log !== nothing
        log_bond_spectrum!(bond_log, epoch, sweep, h, S, keep, d_phys)
    end
    return nothing
end

@inline function _clip_ttn_grad!(G::AbstractArray{<:Real}; maxnorm::Float64 = 10.0)
    gnorm = norm(G)
    if !isfinite(gnorm)
        fill!(G, 0.0)
    elseif gnorm > maxnorm
        G .*= maxnorm / gnorm
    end
    return G
end

function _ttn_run_sweep!(
    tasks::Vector{Tuple{Symbol, Int}},
    ttn::BinaryTTN{T},
    adam::Dict{Int, Any},
    xi_data::AbstractMatrix{Int},
    U::Vector{Matrix{Float64}},
    E_down::Vector{Matrix{Float64}},
    M_up::Vector{Matrix{Float64}},
    η_t::Float64,
    D_max::Int,
    ε_cut::Real,
    sweep::Symbol;
    bond_log = nothing,
    epoch::Int = 0,
    d_phys::Int = 0,
) where T
    N_pad = ttn.N_pad
    psi_all = U[1][:, 1]
    refresh_h = 1
    for (kind, id) in tasks
        if kind === :leaf
            h = _ttn_heap_of_leaf(id, N_pad)
            grad = _ttn_grad_node(ttn, xi_data, h, U, E_down, psi_all, M_up)
            _ttn_adam_leaf!(ttn, adam, h, grad, η_t)
            refresh_h = h
        elseif kind === :internal
            grad = _ttn_grad_node(ttn, xi_data, id, U, E_down, psi_all, M_up)
            _ttn_adam_internal!(ttn, adam, id, grad, η_t)
            refresh_h = id
        elseif kind === :leaf_pair
            update_ttn_leaf_pair!(
                ttn, adam, id, xi_data, U, E_down, psi_all, M_up,
                η_t, D_max, ε_cut;
                sweep = sweep, bond_log = bond_log, epoch = epoch, d_phys = d_phys,
            )
            refresh_h = id
        end
        psi_all = _ttn_refresh_after_node!(U, E_down, M_up, ttn, xi_data, refresh_h)
    end
    return nothing
end

# ─── Training loop ───────────────────────────────────────────────────────────

"""
    train_ttn!(ttn, xi_data, n_epochs, η, D_max; kwargs...) -> nll_hist

Train a `BinaryTTN` Born machine.

Default (`sweep_mode = :zigzag`, `dmrg_pairs = true`):
1. Root-canonicalise.
2. Zigzag bottom-up sweep with incremental environment refresh.
3. Zigzag top-down sweep.
4. NLL estimate.

Use `sweep_mode = :legacy` for the original monotonic schedule with frozen
environments within each half-epoch.

# Keyword arguments (mirror `train_mps!`)
- `verbose`, `nll_samples`   — logging.
- `lr_schedule`              — `f(epoch, n_epochs, η) -> η_t`; `nothing` = constant.
- `val_data`, `val_samples`, `patience`, `val_nll_log` — validation / early stopping.
- `checkpoint_dir`, `checkpoint_every`                 — JLD2 checkpointing.
- `sweep_mode`               — `:zigzag` (default) or `:legacy`.
- `dmrg_pairs`               — DMRG pair updates at leaf-parent nodes (default `true`).
- `ε_cut`                    — SVD truncation floor for leaf-pair updates (default `0`).
- `bond_log`                 — optional bond-spectrum log (like `train_mps!`).
"""
function train_ttn!(ttn       :: BinaryTTN{T},
                    xi_data   :: AbstractMatrix{Int},
                    n_epochs  :: Int,
                    η         :: Real,
                    D_max     :: Int;
                    verbose         :: Bool   = true,
                    nll_samples     :: Int    = 500,
                    lr_schedule                    = nothing,
                    val_data                       = nothing,
                    val_samples     :: Int    = 2_000,
                    patience        :: Int    = typemax(Int),
                    val_nll_log                    = nothing,
                    checkpoint_dir                 = nothing,
                    checkpoint_every               = nothing,
                    sweep_mode      ::Symbol = :zigzag,
                    dmrg_pairs      ::Bool   = true,
                    ε_cut           ::Real   = 0.0,
                    bond_log                       = nothing) where T

    N_pad   = ttn.N_pad
    adam    = Dict{Int, Any}()
    nll_hist = Float64[]

    best_val = Inf; pat_count = 0; stop = false

    verbose && println("train_ttn!: n_sites=$(ttn.n_sites), N_pad=$(N_pad), " *
                       "D_max=$(D_max), epochs=$n_epochs, sweep=$sweep_mode, " *
                       "dmrg_pairs=$dmrg_pairs")

    U      = Vector{Matrix{Float64}}(undef, 2N_pad)
    E_down = Vector{Matrix{Float64}}(undef, 2N_pad)

    leaf_fwd = [j for j in 1:N_pad if ttn.d_vec[j] > 1]
    d_phys   = isempty(leaf_fwd) ? 1 : size(ttn.leaves[leaf_fwd[1]], 1)

    use_zigzag = sweep_mode === :zigzag
    tasks_bu = use_zigzag ?
        _ttn_sweep_tasks(N_pad, leaf_fwd; bottom_up = true,  dmrg_pairs = dmrg_pairs) :
        vcat(
            [( :leaf, j) for j in leaf_fwd],
            [( :internal, h) for h in collect((N_pad - 1):-1:1)],
        )
    tasks_td = use_zigzag ?
        _ttn_sweep_tasks(N_pad, leaf_fwd; bottom_up = false, dmrg_pairs = dmrg_pairs) :
        vcat(
            [( :internal, h) for h in 1:(N_pad - 1)],
            [( :leaf, j) for j in reverse(leaf_fwd)],
        )

    for epoch in 1:n_epochs
        t0  = time()
        η_t = lr_schedule === nothing ? Float64(η) : Float64(lr_schedule(epoch, n_epochs, η))

        root_canonicalize_ttn!(ttn)
        _ttn_upward!(U, ttn, xi_data)
        _ttn_downward!(E_down, ttn, U)
        M_up = _ttn_norm_up(ttn)

        if use_zigzag
            _ttn_run_sweep!(
                tasks_bu, ttn, adam, xi_data, U, E_down, M_up,
                η_t, D_max, ε_cut, :forward;
                bond_log = bond_log, epoch = epoch, d_phys = d_phys,
            )
            root_canonicalize_ttn!(ttn)
            _ttn_upward!(U, ttn, xi_data)
            _ttn_downward!(E_down, ttn, U)
            M_up = _ttn_norm_up(ttn)
            _ttn_run_sweep!(
                tasks_td, ttn, adam, xi_data, U, E_down, M_up,
                η_t, D_max, ε_cut, :backward;
                bond_log = bond_log, epoch = epoch, d_phys = d_phys,
            )
        else
            psi_all = U[1][:, 1]
            for (kind, id) in tasks_bu
                if kind === :leaf
                    h = _ttn_heap_of_leaf(id, N_pad)
                    grad = _ttn_grad_node(ttn, xi_data, h, U, E_down, psi_all, M_up)
                    _ttn_adam_leaf!(ttn, adam, h, grad, η_t)
                else
                    grad = _ttn_grad_node(ttn, xi_data, id, U, E_down, psi_all, M_up)
                    _ttn_adam_internal!(ttn, adam, id, grad, η_t)
                end
            end
            root_canonicalize_ttn!(ttn)
            _ttn_upward!(U, ttn, xi_data)
            _ttn_downward!(E_down, ttn, U)
            psi_all = U[1][:, 1]
            M_up = _ttn_norm_up(ttn)
            for (kind, id) in tasks_td
                if kind === :leaf
                    h = _ttn_heap_of_leaf(id, N_pad)
                    grad = _ttn_grad_node(ttn, xi_data, h, U, E_down, psi_all, M_up)
                    _ttn_adam_leaf!(ttn, adam, h, grad, η_t)
                else
                    grad = _ttn_grad_node(ttn, xi_data, id, U, E_down, psi_all, M_up)
                    _ttn_adam_internal!(ttn, adam, id, grad, η_t)
                end
            end
        end

        root_canonicalize_ttn!(ttn)
        nll = ttn_nll(ttn, xi_data; n_samples = nll_samples)
        push!(nll_hist, nll)

        if verbose
            elapsed = round(time() - t0; digits = 2)
            println("Epoch $epoch/$n_epochs | TTN NLL ≈ $(round(nll; digits=4)) | " *
                    "η=$(round(η_t; sigdigits=3)) | $(elapsed) s")
            _train_log_flush!()
        end

        if val_data !== nothing
            val_nll = ttn_nll(ttn, val_data; n_samples = val_samples)
            val_nll_log !== nothing && push!(val_nll_log, val_nll)
            verbose && (println("  ↳ val NLL ≈ $(round(val_nll; digits=4))  " *
                                "(patience $pat_count/$patience)"); _train_log_flush!())
            if val_nll < best_val
                best_val = val_nll; pat_count = 0
            else
                pat_count += 1
                if pat_count >= patience
                    verbose && (println("Early stopping at epoch $epoch"); _train_log_flush!())
                    stop = true
                end
            end
        end
        stop && break

        if checkpoint_dir !== nothing && checkpoint_every !== nothing &&
                epoch % checkpoint_every == 0
            isdir(checkpoint_dir) || mkpath(checkpoint_dir)
            fn = joinpath(checkpoint_dir, "ttn_epoch_$(lpad(epoch,4,'0')).jld2")
            JLD2.@save fn ttn nll_hist epoch
            verbose && (println("  → checkpoint: $fn"); _train_log_flush!())
        end
    end
    return nll_hist
end

# ─── Ancestral sampling ──────────────────────────────────────────────────────

"""
    sample_ttn(ttn, n_samples; seed=0) -> (paths::Matrix{Float64}, xi::Matrix{Int})

Ancestral sampling from a root-canonical `BinaryTTN` Born machine.

The algorithm samples physical indices leaf-by-leaf (left to right).  For each
leaf `j`, the conditional probability

    P(σ_j | σ_1,…,σ_{j-1}) ∝ A_j[σ_j,:]' M_down_j A_j[σ_j,:]

is computed by walking from the root to leaf `j`, accumulating the downward
norm matrix while using cached upward vectors for already-sampled subtrees and
`M_up = I` (canonical form) for not-yet-sampled ones.

Cost: O(`n_sites` · log(`N_pad`) · χ³) per sample.
"""
function sample_ttn(enc, ttn::BinaryTTN{T}, n_samples::Int; seed::Int = 0) where T

    root_canonicalize_ttn!(ttn)
    N_pad   = ttn.N_pad
    n_sites = ttn.n_sites

    # Pre-build paths from root to each real leaf (constant across samples).
    leaf_paths = Vector{Vector{Int}}(undef, n_sites)
    for j in 1:n_sites
        path = Int[]
        h = _ttn_heap_of_leaf(j, N_pad)
        while h > 1; pushfirst!(path, h); h = h >> 1; end
        pushfirst!(path, 1)
        leaf_paths[j] = path
    end

    # Pre-convert tensors to Float64 once.
    B_f64 = [Float64.(ttn.internals[h]) for h in 1:(N_pad-1)]
    A_f64 = [Float64.(ttn.leaves[j])    for j in 1:N_pad]

    xi_out = zeros(Int, n_samples, n_sites)

    Threads.@threads for s in 1:n_samples
        rng = MersenneTwister(seed + s)

        # M_dn_cache[h] = downward norm matrix accumulated from root to internal
        # node h.  Once computed it is permanently valid for this sample because:
        #   • left-direction steps always use M_sib = I (right subtree unsampled),
        #   • right-direction steps use a rank-1 sibling that is set exactly once.
        M_dn_cache = Vector{Union{Nothing, Matrix{Float64}}}(nothing, 2N_pad)
        M_dn_cache[1] = ones(Float64, 1, 1)

        u_fixed = Dict{Int, Vector{Float64}}()

        for j in 1:n_sites
            h_leaf = _ttn_heap_of_leaf(j, N_pad)
            A      = A_f64[j]                     # (d, χ)
            d      = ttn.d_vec[j]
            path   = leaf_paths[j]

            # Walk path through internal nodes, using/filling the M_dn cache.
            # Second-to-last entry in path is the leaf's parent; last is h_leaf.
            M_dn = M_dn_cache[1]
            for k in 1:(length(path)-2)
                hp = path[k]; hc = path[k+1]
                if !isnothing(M_dn_cache[hc])
                    M_dn = M_dn_cache[hc]   # cache hit
                    continue
                end
                B  = B_f64[hp]
                hl = 2hp
                if hc == hl   # left step: sibling (right) is always unsampled → _I
                    M_dn = _mdown_step_left_I(B, M_dn)
                else           # right step: sibling (left) is always sampled → rank-1
                    M_dn = _mdown_step_right_rank1(B, u_fixed[hl], M_dn)
                end
                M_dn_cache[hc] = M_dn
            end

            # Final step: internal parent → h_leaf (leaf not cached).
            hp = path[end-1]
            B  = B_f64[hp]
            hl = 2hp
            if h_leaf == hl
                leaf_M_dn = _mdown_step_left_I(B, M_dn)
            else
                leaf_M_dn = _mdown_step_right_rank1(B, u_fixed[hl], M_dn)
            end

            # P(σ) ∝ A[σ,:]' leaf_M_dn A[σ,:]
            # Vectorised: AM = A * leaf_M_dn; pv[σ] = dot(AM[σ,:], A[σ,:])
            AM    = A * leaf_M_dn                          # (d, χ) — one DGEMM
            pv    = max.(vec(sum(AM .* A; dims=2)), 0.0)   # (d,)
            s_tot = sum(pv)
            if s_tot < 1e-30; pv .= 1.0 / d; else; pv ./= s_tot; end
            σ_pick = _sample_from_pv!(rng, pv, d)
            xi_out[s, j] = σ_pick

            u_fixed[h_leaf] = A[σ_pick, :]
            _ttn_propagate_up!(ttn, h_leaf, u_fixed)
        end
    end

    paths = Encoders.decode_paths(enc, xi_out)
    return paths, xi_out
end

# Compute the downward norm matrix at h_target, using exact outer-product
# M_up for subtrees in u_fixed and M_I for everything else.
function _ttn_mdown_partial(ttn     :: BinaryTTN{T},
                             h_target:: Int,
                             u_fixed :: Dict{Int,Vector{Float64}},
                             M_I     :: Dict{Int,Matrix{Float64}}) where T
    path = Int[]
    h = h_target
    while h > 1
        pushfirst!(path, h)
        h = _ttn_parent(h)
    end
    pushfirst!(path, 1)

    M_dn = ones(Float64, 1, 1)   # root downward env

    for k in 1:(length(path)-1)
        hp = path[k]; hc = path[k+1]
        B  = Float64.(ttn.internals[hp])
        hl = _ttn_left(hp); hr = _ttn_right(hp)
        if hc == hl   # going left; sibling = hr
            if haskey(u_fixed, hr)
                M_dn = _mdown_step_left_rank1(B, u_fixed[hr], M_dn)
            else
                M_dn = _mdown_step_left_I(B, M_dn)
            end
        else          # going right; sibling = hl
            if haskey(u_fixed, hl)
                M_dn = _mdown_step_right_rank1(B, u_fixed[hl], M_dn)
            else
                M_dn = _mdown_step_right_I(B, M_dn)
            end
        end
    end
    return M_dn
end

# Effective M_up for node h: exact outer product if fixed, else M_I.
function _ttn_eff_mup(h       :: Int,
                       u_fixed :: Dict{Int,Vector{Float64}},
                       M_I     :: Dict{Int,Matrix{Float64}})
    if haskey(u_fixed, h)
        u = u_fixed[h]
        return u * u'
    else
        return M_I[h]
    end
end

# After sampling a leaf, walk up and cache exact upward vectors for ancestors
# whose entire subtree is now sampled.
function _ttn_propagate_up!(ttn::BinaryTTN{T},
                             h_leaf::Int,
                             u_fixed::Dict{Int,Vector{Float64}}) where T
    h = _ttn_parent(h_leaf)
    while h >= 1
        hl = _ttn_left(h); hr = _ttn_right(h)
        if haskey(u_fixed, hl) && haskey(u_fixed, hr)
            B  = Float64.(ttn.internals[h])
            χ_l, χ_r, χ_u = size(B)
            ul = u_fixed[hl]; ur = u_fixed[hr]
            outer = vec(ul * ur')            # col-major kron = ul ⊗ ur
            u_fixed[h] = reshape(B, χ_l * χ_r, χ_u)' * outer
            h == 1 && break
            h = _ttn_parent(h)
        else
            break
        end
    end
end

# ─── Classification (label leaf) ─────────────────────────────────────────────

"""
    class_probabilities_ttn(ttn, xi_path, n_classes) -> Vector{Float64}

Born-rule class probabilities `p(y=c | xi_path)` for a fixed encoded path
(length `ttn.n_sites - 1`).
"""
function class_probabilities_ttn(
    ttn::BinaryTTN{T},
    xi_path::AbstractVector{<:Integer},
    n_classes::Int,
) where {T<:Real}
    @assert length(xi_path) == ttn.n_sites - 1
    amps = Vector{Float64}(undef, n_classes)
    @inbounds for c in 1:n_classes
        x = vcat(collect(Int, xi_path), c)
        amps[c] = abs2(Float64(ttn_amplitude(ttn, x)))
    end
    s = sum(amps)
    if !(s > 0) || !isfinite(s)
        return fill(1.0 / n_classes, n_classes)
    end
    return amps ./ s
end

"""Argmax of `class_probabilities_ttn` (1-based class index)."""
function predict_class_ttn(
    ttn::BinaryTTN{T},
    xi_path::AbstractVector{<:Integer},
    n_classes::Int,
) where {T<:Real}
    return argmax(class_probabilities_ttn(ttn, xi_path, n_classes))
end

"""Fraction of correctly classified rows in `xi_data` (label in last column)."""
function classification_accuracy_ttn(
    ttn::BinaryTTN{T},
    xi_data::AbstractMatrix{<:Integer},
    n_classes::Int,
) where {T<:Real}
    Nd = size(xi_data, 1)
    correct = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:Nd
        xi_row = xi_data[i, :]
        y_true = Int(xi_row[end])
        y_pred = predict_class_ttn(ttn, xi_row[1:(end - 1)], n_classes)
        y_pred == y_true && Threads.atomic_add!(correct, 1)
    end
    return correct[] / Nd
end

# ─── Tree entanglement diagnostics ───────────────────────────────────────────

function _vn_entropy_from_sv(s::AbstractVector{<:Real})
    z = sum(abs2, s)
    if !(isfinite(z)) || z <= 0
        return 0.0
    end
    return sum(-(p/z) * log(p/z) for p in abs2.(s) if p/z > 1e-30)
end

"""
    ttn_subtree_leaves(ttn, h) -> Vector{Int}

Physical leaf indices (1-based time steps) in the subtree rooted at heap node `h`.
"""
function ttn_subtree_leaves(ttn::BinaryTTN, h::Int)::Vector{Int}
    N_pad = ttn.N_pad
    if _ttn_is_leaf(h, N_pad)
        j = _ttn_leaf_idx(h, N_pad)
        return (j <= ttn.n_sites && ttn.d_vec[j] > 1) ? [j] : Int[]
    end
    return vcat(ttn_subtree_leaves(ttn, 2h), ttn_subtree_leaves(ttn, 2h + 1))
end

"""
    TTNInternalCut

Von Neumann entropies at one internal heap node of a root-canonical `BinaryTTN`.
"""
struct TTNInternalCut
    h            :: Int
    depth        :: Int
    χ_l          :: Int
    χ_r          :: Int
    χ_u          :: Int
    leaves_left  :: Vector{Int}
    leaves_right :: Vector{Int}
    S_lr         :: Float64
    S_up         :: Float64
    sv_lr        :: Vector{Float64}
    sv_up        :: Vector{Float64}
end

"""
    ttn_internal_cuts(ttn) -> Vector{TTNInternalCut}

Von Neumann entropies (nats) at every internal node of a root-canonical `BinaryTTN`.

* `S_lr` — entanglement across the **left | right** child bonds at `B_h`.
* `S_up` — entanglement across the **(left ⊗ right) | parent** cut (`χ_u` bond).

`leaves_left` / `leaves_right` are physical site indices in each subtree (tree
grouping, not necessarily contiguous in calendar time).
"""
function ttn_internal_cuts(ttn::BinaryTTN{T}) where {T<:Real}
    root_canonicalize_ttn!(ttn)
    N_pad = ttn.N_pad
    cuts  = TTNInternalCut[]
    for h in 1:(N_pad - 1)
        B = Float64.(ttn.internals[h])
        χ_l, χ_r, χ_u = size(B)
        sv_lr = collect(svd(reshape(B, χ_l, χ_r * χ_u); full = false).S)
        sv_up = collect(svd(reshape(B, χ_l * χ_r, χ_u); full = false).S)
        push!(cuts, TTNInternalCut(
            h, h == 1 ? 0 : Int(floor(log2(h))),
            χ_l, χ_r, χ_u,
            ttn_subtree_leaves(ttn, 2h),
            ttn_subtree_leaves(ttn, 2h + 1),
            _vn_entropy_from_sv(sv_lr), _vn_entropy_from_sv(sv_up),
            sv_lr, sv_up,
        ))
    end
    return cuts
end

"""
    ttn_layer_entropy_summary(cuts) -> (depths, mean_S_lr, mean_S_up, n_nodes)

Mean `S_lr` and `S_up` over internal nodes at each tree depth.
"""
function ttn_layer_entropy_summary(cuts::AbstractVector{TTNInternalCut})
    depths = sort(unique(c.depth for c in cuts))
    mean_lr = Float64[]
    mean_up = Float64[]
    counts  = Int[]
    for d in depths
        grp = [c for c in cuts if c.depth == d]
        n = length(grp)
        push!(mean_lr, sum(c.S_lr for c in grp) / n)
        push!(mean_up, sum(c.S_up for c in grp) / n)
        push!(counts, length(grp))
    end
    return depths, mean_lr, mean_up, counts
end

# Return upward bond dimension for heap node h.
function _ttn_bond_up(ttn::BinaryTTN{T}, h::Int) where T
    N_pad = ttn.N_pad
    if _ttn_is_leaf(h, N_pad)
        return size(ttn.leaves[_ttn_leaf_idx(h, N_pad)], 2)
    else
        return size(ttn.internals[h], 3)
    end
end
