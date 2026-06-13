using Test
using MPSFast
using MPSFast.Encoders
using Random
using LinearAlgebra
using Statistics

# ── Helpers ───────────────────────────────────────────────────────────────────

function random_paths(rng, N, M; σ = 0.02)
    cumsum(randn(rng, N, M) .* σ, dims = 2) .+ 1.0
end

# ── Core: amplitude and partition function ────────────────────────────────────

@testset "mps_amplitude" begin
    rng = MersenneTwister(1)
    M, d, D = 6, 4, 8
    mps = init_mps(M, d, D; rng = rng)

    # Amplitude of every basis state sums to Z in Born measure
    Z_sum = 0.0
    for xi in Iterators.product([1:d for _ in 1:M]...)
        a = mps_amplitude(mps, collect(xi))
        Z_sum += abs2(Float64(a))
    end
    logZ = log_partition_function(mps)
    @test isapprox(Z_sum, exp(logZ); rtol = 1e-4)
end

@testset "left_canonicalize_mps!" begin
    rng = MersenneTwister(2)
    mps = init_mps(8, 3, 6; rng = rng)
    left_canonicalize_mps!(mps)
    # After left canonicalisation, ‖Ψ‖² ≈ 1
    L, _ = norm_environments(mps)
    @test isapprox(L[end][1, 1], 1.0; atol = 1e-5)
end

# ── BasisEncoder ──────────────────────────────────────────────────────────────

@testset "BasisEncoder round-trip" begin
    rng   = MersenneTwister(10)
    paths = random_paths(rng, 200, 10)
    enc   = BasisEncoder(3)
    fit_grid!(enc, paths)
    xi     = encode_paths(enc, paths)
    paths2 = decode_paths(enc, xi)

    @test size(xi) == size(paths)
    @test all(1 .<= xi .<= site_dim(enc))
    # Decode should be close to original up to quantisation error (≤ 1 bucket width)
    bucket_width = (enc.Smax - enc.Smin) / (site_dim(enc) - 1)
    @test maximum(abs.(paths2 .- paths)) <= bucket_width + 1e-10
end

@testset "BasisEncoder chain_length and site_dim" begin
    enc = BasisEncoder(4)
    @test site_dim(enc) == 16
    @test chain_length(enc, 20) == 20
    @test feature_map(enc) === nothing
end

# ── BinaryEncoder ─────────────────────────────────────────────────────────────

@testset "BinaryEncoder round-trip" begin
    rng   = MersenneTwister(11)
    paths = random_paths(rng, 100, 8)
    enc   = BinaryEncoder(3)
    fit_grid!(enc, paths)
    xi     = encode_paths(enc, paths)
    paths2 = decode_paths(enc, xi)

    @test size(xi, 2) == 8 * 3
    @test all(xi .∈ Ref(1:2))
    bucket_width = (enc.Smax - enc.Smin) / (2^enc.m - 1)
    @test maximum(abs.(paths2 .- paths)) <= bucket_width + 1e-10
end

@testset "BinaryEncoder chain_length and site_dim" begin
    enc = BinaryEncoder(4)
    @test site_dim(enc) == 2
    @test chain_length(enc, 10) == 40
end

# ── TrigEncoder ───────────────────────────────────────────────────────────────

@testset "TrigEncoder feature_map" begin
    enc = TrigEncoder(3, 6)
    Φ   = feature_map(enc)
    @test size(Φ) == (2^3, 6)
    # Rows are unit-normalised (each row is a vector of trig values, not normalised
    # but columns should vary smoothly — check no row is all-zero)
    @test all(norm(Φ[k, :]) > 0 for k in axes(Φ, 1))
end

@testset "TrigEncoder encode/decode consistency with BasisEncoder" begin
    rng   = MersenneTwister(12)
    paths = random_paths(rng, 50, 5)
    tb    = BasisEncoder(3)
    tt    = TrigEncoder(3, 4)
    fit_grid!(tb, paths); fit_grid!(tt, paths)
    # Same bucket indices (both use uniform 2^m grid)
    @test encode_paths(tb, paths) == encode_paths(tt, paths)
end

# ── Classification helpers ────────────────────────────────────────────────────

@testset "encode_labeled_paths" begin
    rng    = MersenneTwister(20)
    paths  = random_paths(rng, 30, 5)
    labels = rand(rng, 1:2, 30)
    enc    = BasisEncoder(2)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, labels; n_classes = 2)
    @test size(xi, 2) == classification_chain_length(enc, 5)
    @test all(xi[:, end] .== labels)
end

# ── Training smoke test ───────────────────────────────────────────────────────

@testset "train_mps! smoke (BasisEncoder, 2 epochs)" begin
    rng   = MersenneTwister(42)
    paths = random_paths(rng, 500, 10)
    enc   = BasisEncoder(3)
    fit_grid!(enc, paths)
    xi    = encode_paths(enc, paths)

    M_enc = chain_length(enc, 10)
    mps   = init_mps(M_enc, site_dim(enc), 12; rng = rng)

    nll_hist = train_mps!(mps, xi, 2, 1e-3, 12, 1e-5; verbose = false, nll_samples = 100)

    @test length(nll_hist) == 2
    @test all(isfinite, nll_hist)
end

@testset "train_mps! smoke (TrigEncoder, 2 epochs)" begin
    rng   = MersenneTwister(43)
    paths = random_paths(rng, 300, 6)
    enc   = TrigEncoder(3, 4)
    fit_grid!(enc, paths)
    xi    = encode_paths(enc, paths)
    Phi   = Float32.(feature_map(enc))

    M_enc = chain_length(enc, 6)
    mps   = init_mps(M_enc, site_dim(enc), 8; rng = rng)

    nll_hist = train_mps!(mps, xi, 2, 1e-3, 8, 1e-5;
                          feature_phi = Phi, verbose = false, nll_samples = 50)
    @test length(nll_hist) == 2
    @test all(isfinite, nll_hist)
end

# ── Sampling ──────────────────────────────────────────────────────────────────

@testset "sample_paths reproduces marginal on trained MPS" begin
    rng   = MersenneTwister(50)
    paths = random_paths(rng, 1000, 8)
    enc   = BasisEncoder(3)
    fit_grid!(enc, paths)
    xi    = encode_paths(enc, paths)

    mps   = init_mps(size(xi, 2), site_dim(enc), 16; rng = rng)
    train_mps!(mps, xi, 8, 5e-4, 16, 1e-5; verbose = false, nll_samples = 200)

    sampled, _ = sample_paths(enc, mps, 500; seed = 7)
    @test size(sampled) == (500, 8)
    # Rough marginal check: sampled mean should be within 0.02 of training mean.
    # The paths have σ≈0.02 so this is ~1 standard deviation — meaningful but
    # not brittle given the short training and coarse 8-bin discretisation.
    tr_mean  = mean(paths[:, 1])
    smp_mean = mean(sampled[:, 1])
    @test abs(smp_mean - tr_mean) < 0.02
end

# ── Bipartite entropies ───────────────────────────────────────────────────────

@testset "bipartite_entropies" begin
    rng = MersenneTwister(60)
    mps = init_mps(10, 4, 8; rng = rng)
    Svals, entr = bipartite_entropies(mps)

    @test length(Svals) == 9
    @test length(entr)  == 9
    @test all(isfinite, entr)
    @test all(>=(0), entr)
end

# ── Checkpointing ─────────────────────────────────────────────────────────────

@testset "save_mps_bundle / load_mps_bundle" begin
    rng  = MersenneTwister(70)
    mps  = init_mps(6, 4, 8; rng = rng)
    nll  = [1.0, 0.9, 0.8]
    meta = Dict{String,Any}("test" => true)
    tmp  = tempname() * ".jld2"

    save_mps_bundle(tmp, mps, nll, 3, meta)
    mps2, nll2, epoch2, meta2 = load_mps_bundle(tmp)

    @test epoch2 == 3
    @test nll2 ≈ nll
    @test all(mps[j] ≈ mps2[j] for j in eachindex(mps))

    rm(tmp; force = true)
end
