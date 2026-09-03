#!/usr/bin/env julia
# Validate model-specific entropy / χ bounds on report §8 + random sweeps.

using Pkg
Pkg.activate(dirname(@__DIR__))
using MPSFast
using Printf
using Random

function main()
    println("=== §8 benchmark audit ===")
    model, Δ, B, _ = report_benchmark_model()
    aud = entropy_bounds_audit(model, Δ, B; rng=MersenneTwister(42))
    @printf "S_measured     = %.6f nats\n" aud.S_measured
    @printf "S_fannes       = %.6f (κ=%.4f, F=%.6f, d_cut=%d)\n" aud.S_fannes aud.κ aud.fidelity_diagonal aud.d_cut
    @printf "S_perturbation = %.6f\n" aud.S_perturbation
    @printf "S_hastings     = %.6f (gap=%.4f)\n" aud.S_hastings aud.gap
    @printf "S_combined     = %.6f\n" aud.S_combined
    @printf "ξ measured     = %.4f  (ξ_gap bound = %.4f)\n" aud.correlation_length aud.xi_gap_bound
    @printf "χ measured     = %d  (χ combined = %d)\n" aud.chi_measured aud.chi_combined

    println("\n=== Random sweep (N=4..8, K=1..3, λ=0.5..4) ===")
    rng = MersenneTwister(0)
    n = 0
    n_fail_fannes = 0
    n_fail_combined = 0
    for N in 4:8, K in 1:3, λ in (0.5, 1.0, 2.0, 4.0)
        m, Δr, Br, _ = random_factor_model(N, K; Q=2, factor_strength=λ, rng=rng)
        r = entropy_bounds_audit(m, Δr, Br; rng=rng)
        n += 1
        r.S_measured > r.S_fannes + 1e-8 && (n_fail_fannes += 1)
        r.S_measured > r.S_combined + 1e-6 && (n_fail_combined += 1)
    end
    @printf "instances = %d, Fannes violations = %d, combined violations = %d\n" n n_fail_fannes n_fail_combined
end

main()
