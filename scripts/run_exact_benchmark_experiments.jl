#!/usr/bin/env julia

ENV["GKSwstype"] = "100"

using MPSFast
using LinearAlgebra
using Random
using Printf
using Plots
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_DIR = get(
    ENV,
    "MPSFAST_RESULTS_DIR",
    joinpath(REPO_ROOT, "results", "market_making"),
)
mkpath(OUTPUT_DIR)

model, Δ, B, cores = report_benchmark_model()
H = build_hamiltonian_sparse(model)
vals, vecs = exact_ground_states(H; nev=2, rng=MersenneTwister(42))
E0, E1 = vals
ϕ0 = vecs[1]
dims = site_dims(model)
edges = central_grid_edges(model.Qs)
ϕ0_tensor = lex_vector_to_tensor(ϕ0, dims)

println("[Exact N=8 benchmark]")
@printf "E0 = %.10f\n" E0
@printf "E1 = %.10f\n" E1
@printf "gap = %.10f\n" E1 - E0

entropies = bipartite_entropies_exact(ϕ0_tensor)
open(joinpath(OUTPUT_DIR, "entanglement_entropy.csv"), "w") do io
    println(io, "cut,entropy_nats")
    for (cut, entropy) in enumerate(entropies)
        println(io, "$cut,$entropy")
    end
end

entropy_plot = plot(
    1:length(entropies),
    entropies;
    marker=:circle,
    linewidth=2,
    xlabel="Tensor cut",
    ylabel="Entropy (nats)",
    xticks=1:length(entropies),
    legend=false,
    grid=:y,
    size=(650, 400),
)
savefig(entropy_plot, joinpath(OUTPUT_DIR, "exact_benchmark_entropy.pdf"))
savefig(entropy_plot, joinpath(OUTPUT_DIR, "exact_benchmark_entropy.png"))

χs = [4, 8, 16, 32]
tt_residual = Float64[]
tt_quote_rmse = Float64[]
dmrg_residuals = Float64[]
dmrg_quote_rmse = Float64[]

println("\n[TT-SVD and DMRG convergence]")
@printf "%4s  %12s  %12s  %12s  %12s\n" "chi" "TT residual" "DMRG residual" "TT quote" "DMRG quote"
for χ in χs
    tt_cores, _ = tt_svd(ϕ0_tensor; maxdim=χ)
    ϕtt = align_and_normalize(mps_to_lex_vector(tt_cores, dims), ϕ0)
    Ett = dot(ϕtt, H * ϕtt)
    rtt = norm(H * ϕtt - Ett * ϕtt)
    qtt, _ = quote_log_ratio_errors(ϕtt, ϕ0, model.Qs, edges)

    E, mps, _ = dmrg_ground_state(
        cores;
        maxdim=χ,
        n_sweeps=12,
        cutoff=1e-14,
        rng=MersenneTwister(1000 + χ),
    )
    rdmrg = dmrg_residual(mps, cores, E)
    ϕdmrg = align_and_normalize(
        mps_to_lex_vector([Array(a) for a in mps], dims),
        ϕ0,
    )
    qdmrg, _ = quote_log_ratio_errors(ϕdmrg, ϕ0, model.Qs, edges)

    push!(tt_residual, rtt)
    push!(tt_quote_rmse, qtt)
    push!(dmrg_residuals, rdmrg)
    push!(dmrg_quote_rmse, qdmrg)
    @printf "%4d  %12.4e  %12.4e  %12.4e  %12.4e\n" χ rtt rdmrg qtt qdmrg
end

open(joinpath(OUTPUT_DIR, "benchmark_convergence.csv"), "w") do io
    println(io, "chi,tt_residual,dmrg_residual,tt_quote_rmse,dmrg_quote_rmse")
    for j in eachindex(χs)
        println(
            io,
            join(
                (
                    χs[j],
                    tt_residual[j],
                    dmrg_residuals[j],
                    tt_quote_rmse[j],
                    dmrg_quote_rmse[j],
                ),
                ",",
            ),
        )
    end
end

residual_plot = plot(
    χs,
    tt_residual;
    marker=:circle,
    linewidth=2,
    yscale=:log10,
    xlabel="Bond dimension",
    ylabel="Ritz residual",
    label="TT-SVD",
    xticks=χs,
    grid=true,
)
plot!(
    residual_plot,
    χs,
    dmrg_residuals;
    marker=:square,
    linewidth=2,
    label="DMRG",
)

quote_plot = plot(
    χs,
    tt_quote_rmse;
    marker=:circle,
    linewidth=2,
    yscale=:log10,
    xlabel="Bond dimension",
    ylabel="Quote log-ratio RMSE",
    label="TT-SVD",
    xticks=χs,
    grid=true,
)
plot!(
    quote_plot,
    χs,
    dmrg_quote_rmse;
    marker=:square,
    linewidth=2,
    label="DMRG",
)

convergence_plot = plot(residual_plot, quote_plot; layout=(1, 2), size=(950, 400))
savefig(convergence_plot, joinpath(OUTPUT_DIR, "exact_benchmark_convergence.pdf"))
savefig(convergence_plot, joinpath(OUTPUT_DIR, "exact_benchmark_convergence.png"))

println("\n[Multi-seed DMRG: N=8, chi=16]")
seed_rows = NamedTuple[]
for seed in 1:10
    E, mps, _ = dmrg_ground_state(
        cores;
        maxdim=16,
        n_sweeps=12,
        cutoff=1e-14,
        rng=MersenneTwister(seed),
    )
    residual = dmrg_residual(mps, cores, E)
    ϕdmrg = align_and_normalize(
        mps_to_lex_vector([Array(a) for a in mps], dims),
        ϕ0,
    )
    quote_rmse, _ = quote_log_ratio_errors(ϕdmrg, ϕ0, model.Qs, edges)
    energy_error = abs(E - E0)
    push!(seed_rows, (; seed, energy_error, residual, quote_rmse))
    @printf "seed=%2d  |dE|=%.4e  residual=%.4e  quote RMSE=%.4e\n" seed energy_error residual quote_rmse
end

open(joinpath(OUTPUT_DIR, "multiseed_chi16.csv"), "w") do io
    println(io, "seed,energy_error,residual,quote_rmse")
    for row in seed_rows
        println(io, "$(row.seed),$(row.energy_error),$(row.residual),$(row.quote_rmse)")
    end
end

energy_errors = getproperty.(seed_rows, :energy_error)
seed_residuals = getproperty.(seed_rows, :residual)
seed_quote_rmse = getproperty.(seed_rows, :quote_rmse)
@printf "\nmaximum |dE|       = %.4e\n" maximum(energy_errors)
@printf "maximum residual   = %.4e\n" maximum(seed_residuals)
@printf "maximum quote RMSE = %.4e\n" maximum(seed_quote_rmse)
@printf "relative quote-RMSE spread = %.4e\n" (maximum(seed_quote_rmse) - minimum(seed_quote_rmse)) / mean(seed_quote_rmse)

println("\nResults written to $OUTPUT_DIR")
