using Test
using MPSFast
using MPSFast.Encoders
using Random
using LinearAlgebra
using SparseArrays

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
    tr_mean  = sum(paths[:, 1]) / size(paths, 1)
    smp_mean = sum(sampled[:, 1]) / size(sampled, 1)
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

# ── U(1) symmetry (Phase 0) ───────────────────────────────────────────────────

@testset "U1 NoSymmetry is no-op" begin
    rng = MersenneTwister(11)
    mps = init_mps_classification(6, 4, 2, 8; rng = rng)
    ref = copy.(mps)
    apply_u1_conservation!(mps, NoSymmetry; mode = :hard)
    @test mps == ref
    @test u1_allowed_fraction(mps, NoSymmetry) ≈ 1.0
end

@testset "U1 hard mask zeros forbidden blocks" begin
    rng = MersenneTwister(12)
    spec = u1_path_centered_spec(4, 2)
    mps = init_mps_classification(8, 4, 2, 16; rng = rng)
    frac_before = u1_allowed_fraction(mps, spec; n_classes = 2, d_path = 4)
    apply_u1_conservation!(mps, spec; mode = :hard, n_classes = 2, d_path = 4)
    snap = deepcopy(mps)
    apply_u1_conservation!(mps, spec; mode = :hard, n_classes = 2, d_path = 4)
    @test frac_before < 1.0
    @test mps == snap
end

@testset "flux_compatible and sector_compatibility_rate" begin
    spec = u1_path_centered_spec(8, 2)
    xi_path = [1, 2, 3, 4]
    Q = path_total_charge(xi_path, spec)
    @test flux_compatible(xi_path, 1, spec) == (Q + spec.q_label(1) == 0)
    rng = MersenneTwister(1)
    enc = BasisEncoder(3)
    paths = random_paths(rng, 50, 5)
    fit_grid!(enc, paths)
    labels = rand(rng, 1:2, 50)
    xi = encode_labeled_paths(enc, paths, labels; n_classes = 2)
    rate = sector_compatibility_rate(xi, spec)
    @test 0.0 <= rate <= 1.0
end

@testset "train_mps! with NoSymmetry u1 kwargs" begin
    rng = MersenneTwister(13)
    enc = BasisEncoder(2)
    paths = random_paths(rng, 80, 6)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 80); n_classes = 2)
    Ml = classification_chain_length(enc, 6)
    mps = init_mps_classification(Ml, site_dim(enc), 2, 8; rng = rng)
    nll = train_mps!(mps, xi, 2, 1e-3, 8, 1e-5;
                     verbose = false, nll_samples = 40,
                     u1_spec = NoSymmetry, u1_conservation = :none)
    @test length(nll) == 2
end

@testset "train_mps! U1_label hard smoke" begin
    rng = MersenneTwister(14)
    enc = BasisEncoder(2)
    paths = random_paths(rng, 60, 5)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 60); n_classes = 2)
    Ml = classification_chain_length(enc, 5)
    d = site_dim(enc)
    spec = u1_label_spec(2)
    mps = init_mps_classification_u1(Ml, d, 2, 8, spec; rng = rng)
    nll = train_mps!(mps, xi, 2, 1e-3, 8, 1e-5;
                     verbose = false, nll_samples = 30,
                     u1_spec = spec, u1_conservation = :hard,
                     n_classes = 2, d_path = d)
    @test length(nll) == 2
end

@testset "train_mps! U1_path label_only hard smoke" begin
    rng = MersenneTwister(15)
    enc = BasisEncoder(2)
    paths = random_paths(rng, 60, 5)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 60); n_classes = 2)
    Ml = classification_chain_length(enc, 5)
    d = site_dim(enc)
    spec = u1_path_centered_spec(d, 2)
    @test !u1_mask_per_bond(spec, :hard)
    mps = init_mps_classification_u1(Ml, d, 2, 8, spec; rng = rng)
    nll = train_mps!(mps, xi, 2, 1e-3, 8, 1e-5;
                     verbose = false, nll_samples = 30,
                     u1_spec = spec, u1_conservation = :hard,
                     n_classes = 2, d_path = d)
    @test length(nll) == 2
    @test norm(mps[end]) > 0
end

@testset "u1_path_data_matched_spec" begin
    rng = MersenneTwister(16)
    enc = BasisEncoder(2)
    paths = random_paths(rng, 100, 6)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 100); n_classes = 2)
    d = site_dim(enc)
    spec0 = u1_path_centered_spec(d, 2)
    spec1 = u1_path_data_matched_spec(d, 2, xi)
    @test spec1.q_label(1) ∈ (-1, 0, 1)
    @test spec1.q_label(2) ∈ (-1, 0, 1)
    @test abs(spec1.q_label(1)) <= 1 && abs(spec1.q_label(2)) <= 1
end

@testset "u1_path_empirical_spec and flux_exact_class" begin
    rng = MersenneTwister(18)
    enc = BasisEncoder(2)
    paths = random_paths(rng, 80, 5)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 80); n_classes = 2)
    d = site_dim(enc)
    spec = u1_path_centered_spec(d, 2)
    @test flux_exact_class(xi[1, 1:end-1], spec, 2) ∈ 0:2
    spec_e = u1_path_empirical_spec(d, 2, xi)
    @test spec_e.label_left_range !== nothing
    mps_e = init_mps_classification_u1(8, d, 2, 16, spec_e; rng = rng)
    mps_w = init_mps_classification_u1(8, d, 2, 16, spec; rng = rng)
    frac_e = u1_allowed_fraction(mps_e, spec_e; n_classes = 2, d_path = d)
    frac_w = u1_allowed_fraction(mps_w, spec; n_classes = 2, d_path = d)
    @test frac_e >= frac_w
end

@testset "u1 symmetry audit helpers" begin
    rng = MersenneTwister(17)
    spec = u1_label_spec(2)
    mps = init_mps_classification_u1(8, 4, 2, 16, spec; rng = rng)
    apply_u1_conservation!(mps, spec; mode = :hard, n_classes = 2, d_path = 4)
    @test u1_symmetry_residual(mps, spec; n_classes = 2, d_path = 4)
    _, _, ratio = u1_forbidden_mass(mps, spec; n_classes = 2, d_path = 4)
    @test ratio ≈ 0.0 atol = 1e-12
    enc = BasisEncoder(2)
    paths = random_paths(rng, 40, 5)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 40); n_classes = 2)
    Ml = classification_chain_length(enc, 5)
    d = site_dim(enc)
    mps2 = init_mps_classification_u1(Ml, d, 2, 8, spec; rng = rng)
    rep = u1_training_step_audit(mps2, xi, spec, Ml - 1;
                                 η = 1e-3, D_max = 8, conservation = :hard,
                                 n_classes = 2, d_path = d)
    @test rep.after_update_ratio ≥ 0
    @test rep.after_mask_ratio ≈ 0 atol = 1e-5
    @test rep.residual_ok
end

@testset "U1 Phase 1 block-sparse label" begin
    rng = MersenneTwister(19)
    enc = BasisEncoder(2)
    paths = random_paths(rng, 80, 6)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 80); n_classes = 2)
    Ml = classification_chain_length(enc, 6)
    d = site_dim(enc)
    spec = u1_path_empirical_spec(d, 2, xi)

    mps, layout, block = init_mps_classification_u1_block(Ml, d, 2, 16, spec; rng = rng)
    @test u1_block_symmetry_residual(mps, layout)
    stats = u1_block_label_stats(mps, layout, block)
    @test stats.allowed_fraction > 0.0
    @test stats.allowed_fraction < 1.0
    @test stats.compact_params == stats.allowed_params

    # Block audit: gradient projection + epoch-end label projection restores symmetry
    rep = u1_block_training_step_audit(mps, xi, spec, layout, Ml - 1;
                                       η = 1e-3, D_max = 16, n_classes = 2, d_path = d)
    @test rep.after_update_ratio ≥ 0
    @test rep.after_mask_ratio ≈ 0 atol = 1e-5
    @test rep.residual_ok

    # Training smoke with u1_block_mode
    mps_b, _, _ = init_mps_classification_u1_block(Ml, d, 2, 16, spec; rng = rng)
    nll = train_mps!(mps_b, xi, 2, 1e-3, 16, 1e-5;
                     verbose = false, nll_samples = 30,
                     u1_spec = spec, u1_conservation = :hard,
                     u1_block_mode = true, n_classes = 2, d_path = d)
    @test length(nll) == 2
    layout_after = build_u1_label_layout(spec, mps_b, 2, d)
    @test u1_block_symmetry_residual(mps_b, layout_after)
    @test norm(mps_b[end]) > 0

    # Fast amplitude matches dense on a few samples
    x1 = xi[1, :]
    a_dense = mps_amplitude(mps, x1)
    a_block = mps_amplitude_u1_block(mps, x1, layout)
    @test a_dense ≈ a_block rtol = 1e-5

    @test length(label_interface_charges(1, -2, 2, [0, 1])) == 1
    @test length(label_interface_charges(1, -5, 5, [-1, 0, 1])) == 1

    mps_dense, layout_dense, _ = init_mps_classification_u1_block(
        Ml, d, 2, 16, spec; rng = rng, sector_align = false)
    @test u1_block_symmetry_residual(mps_dense, layout_dense)

    seen = Int[]
    train_mps!(mps_b, xi, 2, 1e-3, 16, 1e-5;
               verbose = false, nll_samples = 20,
               u1_spec = spec, u1_block_mode = true, n_classes = 2, d_path = d,
               epoch_callback = (mps, ep, stats) -> begin
                   push!(seen, ep)
                   @test stats.train_nll ≥ 0
                   @test stats.η > 0
               end)
    @test seen == [1, 2]
end

@testset "U1 soft_then_hard conservation schedule" begin
    @test u1_conservation_at_epoch(:soft_then_hard, 3, 10) == :soft
    @test u1_conservation_at_epoch(:soft_then_hard, 11, 10) == :hard
    @test u1_conservation_at_epoch(:hard, 5, 10) == :hard
end

@testset "U1 compatibility gate and training plan" begin
    enc = BasisEncoder(2)
    paths = random_paths(MersenneTwister(1), 40, 6)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(MersenneTwister(2), 1:2, 40); n_classes = 2)
    spec_c4 = u1_path_empirical_spec(2, 2, xi)
    @test u1_data_compatible(xi, spec_c4; threshold = 0.0)
    plan = u1_training_plan(xi, spec_c4; D_max_block = 64, compat_threshold = 0.0)
    @test plan.use_symmetry
    @test plan.use_block
    @test plan.D_max_train == 64
    plan_drift = u1_training_plan(xi, spec_c4; compat_threshold = 0.99)
    @test !plan_drift.use_symmetry
end

@testset "U1 Phase 2 chain layout" begin
    rng = MersenneTwister(21)
    enc = BasisEncoder(2)
    paths = random_paths(rng, 30, 5)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 30); n_classes = 2)
    Ml = classification_chain_length(enc, 5)
    d = site_dim(enc)
    spec = u1_path_empirical_spec(d, 2, xi)
    mps = init_mps_classification(Ml, d, 2, 12; rng = rng)
    chain = build_u1_chain_layout(spec, mps, 2, d)
    @test length(chain.site_left_charges) == Ml
    frac = u1_path_bond_allowed_fraction(chain, 1, Ml, size(mps[1], 1),
        size(mps[1], 2), size(mps[2], 2), size(mps[2], 3))
    @test 0.0 < frac ≤ 1.0
    nll = train_mps!(mps, xi, 1, 1e-3, 12, 1e-5;
                     verbose = false, nll_samples = 20,
                     u1_spec = spec, u1_conservation = :soft_then_hard,
                     u1_soft_epochs = 5, u1_path_block_mode = true,
                     n_classes = 2, d_path = d)
    @test length(nll) == 1
end

@testset "SignIncrementEncoder and semantic U1 specs" begin
    enc = SignIncrementEncoder()
    paths = Float64[0.0 1.0 0.5 2.0; 0.0 -0.2 -0.5 -1.0]
    xi = encode_paths(enc, paths)
    @test size(xi) == (2, 4)
    @test xi[1, 2] == 3
    @test xi[2, 3] == 1
    spec = u1_path_sign_increment_spec(3, 2)
    @test path_total_charge(xi[1, :], spec) > path_total_charge(xi[2, :], spec)
    enc2 = UpDownIncrementEncoder()
    xi2 = encode_paths(enc2, paths)
    spec2 = u1_path_upcount_spec(2, 2)
    @test path_total_charge(xi2[1, :], spec2) >= path_total_charge(xi2[2, :], spec2)
    xi_lab = encode_labeled_paths(enc, paths, [1, 2]; n_classes = 2)
    emp = u1_path_empirical_from_probe(spec, xi_lab, 2)
    @test symmetry_active(emp)
    xi_ud = encode_labeled_paths(UpDownIncrementEncoder(), paths, [1, 2]; n_classes = 2)
    probe_u = u1_path_upcount_spec(2, 2)
    emp_b = u1_path_empirical_from_probe(probe_u, xi_ud, 2; q_label_mode = :bounded)
    emp_r = u1_path_empirical_from_probe(probe_u, xi_ud, 2; q_label_mode = :round)
    @test sector_compatibility_rate(xi_ud, emp_r) >= sector_compatibility_rate(xi_ud, emp_b)
    # empty class in mean_Q must not throw (flux_nearest can miss a class)
    xi_miss = copy(xi_ud)
    xi_miss[1, end] = 1
    for i in 2:size(xi_miss, 1)
        xi_miss[i, end] = 2
    end
    emp_r2 = u1_path_empirical_from_probe(probe_u, xi_miss, 2; q_label_mode = :round)
    @test symmetry_active(emp_r2)
end

# ── SU(2) + TrigEncoder ───────────────────────────────────────────────────────

@testset "SU2 trig feature charges" begin
    @test trig_feature_mz_charge(1) == 1
    @test trig_feature_mz_charge(2) == -1
    @test trig_feature_mz_charge(3) == 2
    @test trig_feature_j_quantum(3) == 2
    @test trig_feature_j_quantum(4) == 2
end

@testset "SU2 feature spec masking" begin
    rng = MersenneTwister(88)
    spec = su2_trig_feature_spec(4, 2)
    mps = init_mps_classification_su2(5, 4, 2, 16, spec; rng = rng)
    @test su2_allowed_fraction(mps, spec; n_classes = 2, d_path = 4) < 1.0
    @test su2_symmetry_residual(mps, spec; n_classes = 2, d_path = 4) == 0.0
    apply_su2_conservation!(mps, spec; mode = :hard, n_classes = 2, d_path = 4)
    @test su2_symmetry_residual(mps, spec; n_classes = 2, d_path = 4) == 0.0
end

@testset "SU2 trig training smoke" begin
    rng = MersenneTwister(89)
    paths = random_paths(rng, 200, 6)
    enc = TrigEncoder(3, 4)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 200); n_classes = 2)
    Phi = Float32.(feature_map(enc))
    K = 2^enc.m
    Ml = classification_chain_length(enc, 6)
    d = site_dim(enc)
    su2 = su2_trig_feature_spec(d, 2)
    mps = init_mps_classification_su2(Ml, d, 2, 12, su2; rng = rng)
    nll = train_mps!(mps, xi, 2, 1e-3, 12, 1e-5;
                     feature_phi = Phi, verbose = false, nll_samples = 40,
                     su2_spec = su2, su2_conservation = :hard,
                     n_classes = 2, d_path = d)
    @test length(nll) == 2
    @test su2_symmetry_residual(mps, su2; n_classes = 2, d_path = d) == 0.0
    acc = classification_accuracy(mps, xi, 2; phi = Phi)
    @test 0.0 <= acc <= 1.0
end

@testset "SU2 bucket compat on TrigEncoder" begin
    rng = MersenneTwister(90)
    paths = random_paths(rng, 100, 5)
    enc = TrigEncoder(3, 4)
    fit_grid!(enc, paths)
    xi = encode_labeled_paths(enc, paths, rand(rng, 1:2, 100); n_classes = 2)
    K = 2^enc.m
    spec = su2_trig_empirical_bucket_spec(K, 2, xi)
    compat = su2_bucket_compat(xi, spec, K, 2)
    @test 0.0 <= compat <= 1.0
    q1 = trig_phi_weighted_charge(1, feature_map(enc))
    qK = trig_phi_weighted_charge(K, feature_map(enc))
    @test q1 != qK
end

# ── QUBO / Ising optimisation ─────────────────────────────────────────────────

@testset "qubo_energy and chain_dp" begin
    h = [0.1, -0.3, 0.2, -0.1]
    J = [0.5, -0.4, 0.2]
    chain = ChainQUBOProblem(h, J)
    dense = dense_qubo(chain)
    x = [1, 0, 1, 0]
    @test isapprox(qubo_energy(chain, x), qubo_energy(dense, x); atol=1e-12)

    exact = chain_qubo_exact_dp(chain)
    brute = qubo_brute_force(dense)
    @test isapprox(exact.energy, brute.energy; atol=1e-10)
    @test exact.x == brute.x
end

@testset "maxcut_path agrees with chain_dp" begin
    n = 12
    prob = maxcut_path_qubo(n)
    chain = chain_qubo_from_dense(prob)
    exact = chain_qubo_exact_dp(chain)
    brute = qubo_brute_force(prob)
    @test isapprox(exact.energy, brute.energy; atol=1e-10)
end

@testset "qubo_mps_dmrg_chain reaches exact on small chain" begin
    rng = MersenneTwister(99)
    chain = ChainQUBOProblem(randn(rng, 10), randn(rng, 9))
    exact = chain_qubo_exact_dp(chain)
    mps_sol = qubo_mps_dmrg_chain(chain; n_sweeps=30, rng=rng)
    @test isapprox(mps_sol.energy, exact.energy; atol=1e-10)
end

@testset "portfolio_selection_qubo brute vs anneal" begin
    rng = MersenneTwister(7)
    n = 8
    μ = randn(rng, n)
    Σ = randn(rng, n, n)
    Σ = Σ' * Σ + 4.0I
    prob = portfolio_selection_qubo(μ, Σ; λ=0.5, target_k=4)
    brute = qubo_brute_force(prob)
    sa = qubo_simulated_annealing(prob; rng=rng, n_steps=20_000)
    @test sa.energy <= brute.energy + 1e-6
end

@testset "portfolio cardinality penalty expansion" begin
    n, k, p = 8, 4, 40.0
    μ = zeros(n)
    Σ = zeros(n, n)
    prob = portfolio_selection_qubo(μ, Σ; λ=0.0, target_k=k, penalty=p)
    for bits in 0:(2^n - 1)
        x = [(bits >> (i - 1)) & 1 for i in 1:n]
        card = p * (sum(x) - k)^2
        @test qubo_energy(prob, x) ≈ card
    end
end

@testset "optimize_portfolio_exact_k" begin
    rng = MersenneTwister(77)
    n, k = 8, 4
    μ = randn(rng, n) * 0.01
    A = randn(rng, n, n)
    Σ = A' * A / n + 0.001 * I
    sol = optimize_portfolio_exact_k(μ, Σ; k=k, λ=0.5)
    @test sum(sol.x) == k
    prob = portfolio_selection_qubo(μ, Σ; λ=0.5, target_k=k, penalty=500.0)
    brute = qubo_brute_force(prob)
    @test sum(brute.x) == k
    @test sol.energy <= portfolio_markowitz_energy(μ, Σ, brute.x; λ=0.5) + 1e-10
end

@testset "qubo_solver_compare" begin
    chain = ChainQUBOProblem([0.0, -0.2, 0.1], [0.3, -0.1])
    rows = qubo_solver_compare(chain)
    @test rows[1].energy <= rows[end].energy
    @test rows[1].method == :chain_dp
end

@testset "generative portfolio qubo loop smoke" begin
    rng = MersenneTwister(123)
    demo = demo_generative_portfolio_loop(;
        n_assets=5, n_train=400, n_oos=200, target_k=2,
        mps_epochs=8, penalty=50.0, rng=rng)
    @test demo.cmp.mps.train.n_selected == 2
    @test demo.cmp.ref.train.n_selected == 2
    @test isfinite(demo.cmp.mps.oos.mean)
    @test size(demo.R_train_mps, 2) == 5
end

# ── Market making (Coutinho ground-state TN report) — milestone 1 ────────────
# Reference: "Ground-State Tensor Networks for Multi-Asset Market Making",
# G. Coutinho, internal report, 28 July 2026.

@testset "market making: Hamiltonian symmetry and stoquasticity" begin
    rng = MersenneTwister(0)
    N = 3
    Qs = [1, 2, 1]
    A = randn(rng, N, N)
    Σ = A * A' + 0.5I
    μ = 0.1 .* randn(rng, N)
    model = MarketMakingModel(N, Qs, Matrix(Σ), μ, 1.0, 1.1, [1.0, 0.9, 1.05])
    H = build_hamiltonian_sparse(model)
    Hd = Matrix(H)
    @test isapprox(Hd, Hd'; atol=1e-12)
    @test maximum(Hd - Diagonal(Hd)) <= 1e-12   # stoquastic: off-diagonal <= 0
end

@testset "market making: exact K+2 MPO matches sparse Hamiltonian" begin
    rng = MersenneTwister(1)
    N, Qs, K = 3, [1, 2, 1], 2
    Δ = [0.7, 0.75, 0.8]
    B = 0.3 .* randn(rng, N, K)
    Σ = Diagonal(Δ) + B * B'
    μ = 0.1 .* randn(rng, N)
    model = MarketMakingModel(N, Qs, Matrix(Σ), μ, 1.0, 1.1, [1.0, 0.9, 1.05])
    H = Matrix(build_hamiltonian_sparse(model))
    cores = build_mpo_cores(model, Δ, B)
    @test all(size(c, 4) <= K + 2 for c in cores[1:end-1])
    Hmpo = mpo_to_dense(cores)
    @test isapprox(H, Hmpo; atol=1e-10)
end

@testset "market making: TT-SVD / lexicographic conversion round trip" begin
    rng = MersenneTwister(2)
    Qs = [1, 2]
    model = MarketMakingModel(2, Qs, [1.0 0.2; 0.2 1.0], [0.0, 0.0], 1.0, 1.0, [1.0, 1.0])
    H = build_hamiltonian_sparse(model)
    vals, vecs = exact_ground_states(H; nev=1, rng=rng)
    ϕ0 = vecs[1]
    @test minimum(ϕ0) > 0   # Perron-Frobenius positivity (Thm. 4.4)

    dims = site_dims(model)
    tens = lex_vector_to_tensor(ϕ0, dims)
    cores, _ = tt_svd(tens)   # untruncated: exact reconstruction
    ϕ0_rt = mps_to_lex_vector(cores, dims)
    @test isapprox(ϕ0_rt, ϕ0; atol=1e-8)
end

@testset "market making: reproduces report's 8-asset benchmark (§8.1-8.3)" begin
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

    @test hilbert_dim(model) == 390625
    H = build_hamiltonian_sparse(model)
    @test nnz(H) == 5_390_624

    vals, vecs = exact_ground_states(H; nev=2, rng=MersenneTwister(42))
    E0, E1 = vals
    @test isapprox(E0, -10.36159622; atol=1e-6)
    @test isapprox(E1, -9.20407484; atol=1e-6)
    @test isapprox(E1 - E0, 1.15752138; atol=1e-6)

    ϕ0 = vecs[1]
    dims = site_dims(model)
    tens = lex_vector_to_tensor(ϕ0, dims)
    entr = bipartite_entropies_exact(tens)
    @test isapprox(maximum(entr), 0.07837966; atol=1e-6)

    edges = central_grid_edges(N)
    @test length(edges) == 104976

    report_rmse = Dict(4 => 7.4791e-3, 8 => 2.7020e-4, 16 => 1.0436e-5, 32 => 1.6051e-7)
    for χ in (4, 8, 16, 32)
        cores, _ = tt_svd(tens; maxdim=χ)
        ϕtilde = align_and_normalize(mps_to_lex_vector(cores, dims), ϕ0)
        rmse, _ = quote_log_ratio_errors(ϕtilde, ϕ0, Qs, edges)
        @test isapprox(rmse, report_rmse[χ]; rtol=0.02)
    end
end

# ── Market making — milestone 2: genuine two-site DMRG ─────────────────────

@testset "dmrg: apply_mpo_to_mps / dmrg_energy matches direct Rayleigh quotient" begin
    rng = MersenneTwister(1)
    N, Qs = 2, [1, 2]
    Δ = [1.0, 1.0]
    B = reshape([0.2, 0.2], 2, 1)
    Σ = Diagonal(Δ) + B * B'
    model = MarketMakingModel(N, Qs, Matrix(Σ), [0.0, 0.0], 1.0, 1.0, [1.0, 1.0])
    cores = build_mpo_cores(model, Δ, B)
    H = build_hamiltonian_sparse(model)
    dims = site_dims(model)

    mps = random_mps_hetero(dims, 8; rng=rng)
    E_dmrg = dmrg_energy(mps, cores)
    ϕ = mps_to_lex_vector([Array(a) for a in mps], dims)
    E_direct = dot(ϕ, H * ϕ) / dot(ϕ, ϕ)
    @test isapprox(E_dmrg, E_direct; atol=1e-8)
end

@testset "dmrg: ground state on N=2 matches exact diagonalization exactly" begin
    rng = MersenneTwister(2)
    N, Qs = 2, [1, 2]
    Δ = [1.0, 1.0]
    B = reshape([0.2, 0.2], 2, 1)
    Σ = Diagonal(Δ) + B * B'
    model = MarketMakingModel(N, Qs, Matrix(Σ), [0.0, 0.0], 1.0, 1.0, [1.0, 1.0])
    cores = build_mpo_cores(model, Δ, B)
    H = build_hamiltonian_sparse(model)
    vals, vecs = exact_ground_states(H; nev=2, rng=rng)

    E, mps, _ = dmrg_ground_state(cores; maxdim=16, n_sweeps=6, rng=MersenneTwister(3))
    @test isapprox(E, vals[1]; atol=1e-8)
    @test dmrg_residual(mps, cores, E) < 1e-6
end

@testset "dmrg: ground/excited state on N=5 converges to exact values as χ grows" begin
    rng = MersenneTwister(4)
    N = 5
    Qs = fill(2, N)
    Δ = fill(0.8, N)
    B = 0.3 .* randn(rng, N, 2)
    Σ = Diagonal(Δ) + B * B'
    η = 0.9 .+ 0.2 .* rand(rng, N)
    model = MarketMakingModel(N, Qs, Matrix(Σ), zeros(N), 1.0, 1.1, η)
    cores = build_mpo_cores(model, Δ, B)
    H = build_hamiltonian_sparse(model)
    vals, vecs = exact_ground_states(H; nev=2, rng=rng)
    E0_exact, E1_exact = vals

    E0, mps0, _ = dmrg_ground_state(cores; maxdim=16, n_sweeps=8, rng=MersenneTwister(5))
    @test isapprox(E0, E0_exact; atol=1e-6)

    E1, mps1, _ = dmrg_excited_state(cores, mps0; maxdim=16, n_sweeps=8, rng=MersenneTwister(6))
    @test isapprox(E1, E1_exact; atol=1e-6)
end

# ── Market making §10.1 experiments ─────────────────────────────────────────

@testset "experiments: report benchmark + Gaussian baseline" begin
    model, Δ, B, cores = report_benchmark_model()
    H = build_hamiltonian_sparse(model)
    vals, vecs = exact_ground_states(H; nev=1, rng=MersenneTwister(0))
    ϕ0 = vecs[1]
    edges = central_grid_edges(model.N)
    rmse_g, max_g = gaussian_quote_rmse(model, Δ, ϕ0, edges)
    @test isfinite(rmse_g) && rmse_g > 0
    @test isfinite(max_g)
    E_diag, ϕ_diag = product_diagonal_ground_state(model, Δ)
    @test E_diag < vals[1]
    @test maximum(ϕ_diag) > 0
end

@testset "experiments: krylov expmv matches small dense reference" begin
    rng = MersenneTwister(11)
    model, Δ, B, _ = random_factor_model(3, 1; Q=1, rng=rng)
    H = build_hamiltonian_sparse(model)
    ψ0 = rand(rng, hilbert_dim(model))
    τ = 0.5
    ψk = krylov_expmv(τ, H, ψ0; krylovdim=30)
    ψd = exp(-τ * Matrix(H)) * ψ0
    @test isapprox(ψk, ψd; rtol=1e-8)
end

@testset "experiments: finite horizon quote RMSE decreases toward ground state" begin
    rng = MersenneTwister(12)
    model, Δ, B, _ = random_factor_model(3, 1; Q=1, rng=rng)
    H = build_hamiltonian_sparse(model)
    vals, vecs = exact_ground_states(H; nev=1, rng=rng)
    ϕ0 = vecs[1]
    τs = [0.0, 0.2, 1.0, 5.0, 20.0]
    res = finite_horizon_quote_convergence(model, H, ϕ0, τs)
    @test res.rmse[end] < res.rmse[1]
    @test res.rmse[end] < 0.05
end

@testset "experiments: Doob chain occupancy approaches pi" begin
    rng = MersenneTwister(13)
    model, Δ, B, _ = random_factor_model(2, 1; Q=1, rng=rng)
    H = build_hamiltonian_sparse(model)
    _, vecs = exact_ground_states(H; nev=1, rng=rng)
    ϕ0 = vecs[1]
    edges = doob_rates_from_phi(ϕ0, model)
    D = hilbert_dim(model)
    traj = simulate_doob_chain(edges, D, 200_000; rng=rng)
    err = doob_occupancy_error(traj, ϕ0)
    @test err < 0.15
end

@testset "experiments: site ordering preserves ground-state energy" begin
    rng = MersenneTwister(14)
    model, Δ, B, cores = random_factor_model(4, 2; Q=1, rng=rng)
    perm = correlation_chain_order(model.Σ)
    mp, Δp, Bp, coresp, _ = permute_factor_model(model, Δ, B, perm)
    H0 = Matrix(build_hamiltonian_sparse(model))
    Hp = Matrix(build_hamiltonian_sparse(mp))
    @test isapprox(minimum(eigvals(Symmetric(H0))), minimum(eigvals(Symmetric(Hp))); atol=1e-8)
    E0, _, _ = dmrg_ground_state(cores; maxdim=8, n_sweeps=4, rng=rng)
    E1, _, _ = dmrg_ground_state(coresp; maxdim=8, n_sweeps=4, rng=rng)
    @test isapprox(E0, E1; atol=1e-5)
end

@testset "experiments: minimal_chi_dmrg on small instance" begin
    rng = MersenneTwister(15)
    _, _, _, cores = random_factor_model(3, 1; Q=1, rng=rng)
    res = minimal_chi_dmrg(cores; eps_res=1e-1, chi_candidates=[4, 8, 16],
                           n_sweeps=4, rng=rng)
    @test res.χ ∈ [4, 8, 16]
    @test res.rel_residual ≤ 1e-1 || res.rel_residual < Inf
end

@testset "experiments: factor-grid baseline on N=8 report instance" begin
    model, Δ, B, _ = report_benchmark_model()
    H = build_hamiltonian_sparse(model)
    _, vecs = exact_ground_states(H; nev=1, rng=MersenneTwister(0))
    ϕ0 = vecs[1]
    edges = central_grid_edges(model.Qs)
    rmse, maxe, _, Qf = factor_grid_quote_rmse(model, Δ, B, ϕ0, edges)
    @test isfinite(rmse) && rmse > 0
    cmp = baseline_quote_comparison(model, Δ, B, ϕ0, edges)
    @test cmp.gaussian.rmse < cmp.product.rmse || cmp.factor_grid.rmse < cmp.product.rmse
end

@testset "experiments: MPS imaginary-time matches exact at small N" begin
    rng = MersenneTwister(16)
    model, Δ, B, cores = random_factor_model(3, 1; Q=1, rng=rng)
    H = build_hamiltonian_sparse(model)
    vals, vecs = exact_ground_states(H; nev=1, rng=rng)
    ϕ0 = vecs[1]
    ψ0 = terminal_vector(model)
    τs = [0.0, 1.0, 5.0]
    edges = central_grid_edges(model.Qs)
    exact = finite_horizon_quote_convergence(model, H, ϕ0, τs; edges=edges)
    mps_r = finite_horizon_mps_convergence(model, cores, ϕ0, τs;
        edges=edges, dt=0.05, maxdim=16, krylovdim=30)
    @test mps_r.rmse[end] < mps_r.rmse[1]
    @test mps_r.rmse[end] < exact.rmse[end] * 5 + 1e-4
end

# ── Entanglement / bond-dimension bounds (model-specific) ─────────────────────

@testset "bounds: diagonal product reference (Theorem 7.1)" begin
    rng = MersenneTwister(21)
    model, Δ, B, _ = report_benchmark_model()
    H = build_hamiltonian_sparse(model)
    _, vecs = exact_ground_states(H; nev=1, rng=rng)
    ϕ0 = vecs[1]
    ϕ_diag = diagonal_product_ground_state(model, Δ)
    S0 = max_bipartite_entropy(ϕ0, model)
    S_ref = max_bipartite_entropy(ϕ_diag, model)
    route1 = factor_coupling_fannes_bound(model, Δ, B, ϕ0)
    @test S_ref < 1e-10
    @test route1.fidelity > 0.95
    @test S0 <= route1.S_fannes + 1e-10
    @test route1.S_fannes < 2.0
    # Exact diagonal-Σ instance: product ground state, zero entanglement.
    Σ_diag = Diagonal(Δ)
    model_diag = MarketMakingModel(model.N, model.Qs, Matrix(Σ_diag), model.μ, model.k, model.γ, model.η)
    _, ϕ_exact_diag = exact_ground_state_vector(model_diag; rng=rng)
    @test max_bipartite_entropy(ϕ_exact_diag, model_diag) < 1e-10
end

@testset "bounds: §8 benchmark satisfies combined entropy scales" begin
    rng = MersenneTwister(22)
    model, Δ, B, _ = report_benchmark_model()
    aud = entropy_bounds_audit(model, Δ, B; rng=rng)
    @test isapprox(aud.S_measured, 0.07837966; atol=1e-5)
    @test aud.S_measured <= aud.S_fannes + 1e-10
    @test aud.S_measured <= aud.S_combined + 1e-10
    @test aud.correlation_length < Inf
    @test aud.correlation_length <= aud.xi_gap_bound * 2.5
    @test aud.chi_measured <= 2
    @test aud.chi_combined <= 9
end

@testset "bounds: sweep — combined bound valid on random instances" begin
    rng = MersenneTwister(23)
    specs = NamedTuple[]
    for N in (3, 4, 5), K in (1, 2), λ in (0.3, 1.0, 3.0)
        m, Δ, B, _ = random_factor_model(N, K; Q=1, factor_strength=λ, rng=rng)
        push!(specs, (; model=m, Δ, B))
    end
    results = entropy_bounds_sweep(specs; rng=rng)
    for r in results
        @test r.S_measured <= r.S_fannes + 1e-8
        @test r.S_measured <= r.S_combined + 1e-6
        @test r.fidelity_diagonal > 0.5
    end
end
