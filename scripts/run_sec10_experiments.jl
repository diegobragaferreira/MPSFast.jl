#!/usr/bin/env julia
# Coutinho §10.1 experiments — milestones 3–4 (phase diagram, MPS imtime, baselines).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MPSFast
using LinearAlgebra
using SparseArrays
using Random
using Printf
using Dates

rng = MersenneTwister(42)

println("=" ^ 72)
println("§10.1 experiments — $(Dates.now())")
println("=" ^ 72)

# ── 1. Baselines (N=8) ─────────────────────────────────────────────────────
model, Δ, B, cores = report_benchmark_model()
H = build_hamiltonian_sparse(model)
vals, vecs = exact_ground_states(H; nev=2, rng=rng)
E0, E1 = vals
ϕ0 = vecs[1]
ΔH = E1 - E0
edges = central_grid_edges(model.Qs)
cmp = baseline_quote_comparison(model, Δ, B, ϕ0, edges)

println("\n[Baselines N=8 — quote RMSE]")
@printf "  product (diag Σ)  = %.4e\n" cmp.product.rmse
@printf "  Gaussian harmonic = %.4e\n" cmp.gaussian.rmse
@printf "  factor-grid       = %.4e\n" cmp.factor_grid.rmse

# ── 2. Finite horizon: exact Krylov vs MPS TEBD ─────────────────────────────
τs = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
exact_fh = finite_horizon_quote_convergence(model, H, ϕ0, τs)
mps_fh = finite_horizon_mps_convergence(model, cores, ϕ0, τs; dt=0.05, maxdim=32)
println("\n[Finite horizon N=8 — quote RMSE vs φ₀]")
@printf "  %6s  %12s  %12s\n" "τ" "exact" "MPS"
for (τ, re, rm) in zip(τs, exact_fh.rmse, mps_fh.rmse)
    @printf "  %6.1f  %12.4e  %12.4e\n" τ re rm
end

# ── 3. Phase diagram (full grid; use reduced sweeps for N≥50) ─────────────────
println("\n[Phase diagram — minimal χ, ε_res=1e-2]")
specs = default_phase_diagram_specs()
for (i, spec) in enumerate(specs)
    N, K, Q, fs = spec
    sweeps = N >= 50 ? 4 : 6
    chis = N >= 50 ? [4, 8, 16, 24, 32, 48, 64] : [4, 8, 16, 24, 32, 48, 64, 96, 128]
    pt = phase_diagram_point(N, K; Q=Q, factor_strength=fs, eps_res=1e-2,
        chi_candidates=chis, n_sweeps=sweeps, rng=MersenneTwister(1000 + i))
    @printf("  N=%3d K=%d d=%2d λ=%.1f  χ_ε=%3d  rel=%.2e  E0=%.5f\n",
        N, K, (2Q + 1), fs, pt.chi, pt.rel_residual, pt.E0)
    flush(stdout)
end

println("\nDone.")
