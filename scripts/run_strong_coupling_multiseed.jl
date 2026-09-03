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

function strong_coupling_model(; seed::Int=150)
    N, K, Q = 50, 3, 2
    coupling_ratio = 10.0
    rng = MersenneTwister(seed)
    Δ = 0.6 .+ 0.3 .* rand(rng, N)
    η = 0.9 .+ 0.2 .* rand(rng, N)
    Braw = randn(rng, N, K)
    B = sqrt(coupling_ratio * minimum(Δ)) .* Braw ./ opnorm(Braw)
    Σ = Diagonal(Δ) + B * B'
    model = MarketMakingModel(N, fill(Q, N), Matrix(Σ), zeros(N), 1.0, 1.1, η)
    cores = build_mpo_cores(model, Δ, B)
    actual_ratio = opnorm(B)^2 / minimum(Δ)
    return model, cores, actual_ratio
end

function sampled_edges(N::Int, count::Int; seed::Int=31_415)
    rng = MersenneTwister(seed)
    edges = Tuple{Vector{Int},Vector{Int}}[]
    for _ in 1:count
        q = rand(rng, -1:1, N)
        qn = copy(q)
        i = rand(rng, 1:N)
        qn[i] += rand(rng, Bool) ? 1 : -1
        push!(edges, (q, qn))
    end
    return edges
end

function mps_log_ratio(mps, q::Vector{Int}, qn::Vector{Int}, Qs::Vector{Int})
    x = q .+ Qs .+ 1
    xn = qn .+ Qs .+ 1
    a = abs(mps_amplitude(mps, x))
    an = abs(mps_amplitude(mps, xn))
    return log(max(a, floatmin(Float64))) - log(max(an, floatmin(Float64)))
end

function quote_rmse(mps, reference, edges, Qs)
    squared_error = 0.0
    for (q, qn) in edges
        difference =
            mps_log_ratio(mps, q, qn, Qs) -
            mps_log_ratio(reference, q, qn, Qs)
        squared_error += difference^2
    end
    return sqrt(squared_error / length(edges))
end

model, cores, actual_ratio = strong_coupling_model()
edges = sampled_edges(model.N, 500)
run_specs = [
    (χ=32, seeds=1:5),
    (χ=64, seeds=1:3),
    (χ=96, seeds=1:1),
]
n_sweeps = 8
solutions = NamedTuple[]

println("[Strong-coupling multi-seed experiment]")
@printf "N=%d K=%d d=%d rho=%.1f sweeps=%d sampled_edges=%d\n" model.N 3 5 actual_ratio n_sweeps length(edges)

for spec in run_specs
    for seed in spec.seeds
        elapsed = @elapsed begin
            E, mps, history = dmrg_ground_state(
                cores;
                maxdim=spec.χ,
                n_sweeps=n_sweeps,
                cutoff=1e-13,
                rng=MersenneTwister(100_000 + 1_000 * spec.χ + seed),
            )
        end
        residual = dmrg_residual(mps, cores, E)
        scaled_residual = residual / max(abs(E), 1.0)
        push!(
            solutions,
            (;
                χ=spec.χ,
                seed,
                E,
                residual,
                scaled_residual,
                elapsed,
                final_local_energy_change=abs(history[end] - history[end - 1]),
                mps,
            ),
        )
        @printf(
            "chi=%2d seed=%d E=%.10f residual=%.4e scaled=%.4e time=%.1fs\n",
            spec.χ,
            seed,
            E,
            residual,
            scaled_residual,
            elapsed,
        )
        flush(stdout)
    end
end

best_index = argmin(getproperty.(solutions, :E))
reference = solutions[best_index]
rows = NamedTuple[]
for solution in solutions
    policy_rmse = quote_rmse(solution.mps, reference.mps, edges, model.Qs)
    push!(
        rows,
        (;
            χ=solution.χ,
            seed=solution.seed,
            E=solution.E,
            energy_above_best=solution.E - reference.E,
            residual=solution.residual,
            scaled_residual=solution.scaled_residual,
            quote_rmse=policy_rmse,
            elapsed=solution.elapsed,
            final_local_energy_change=solution.final_local_energy_change,
        ),
    )
end

open(joinpath(OUTPUT_DIR, "strong_coupling_multiseed.csv"), "w") do io
    println(
        io,
        "chi,seed,E,energy_above_best,residual,scaled_residual,quote_rmse,elapsed_seconds,final_local_energy_change",
    )
    for r in rows
        println(
            io,
            join(
                (
                    r.χ,
                    r.seed,
                    r.E,
                    r.energy_above_best,
                    r.residual,
                    r.scaled_residual,
                    r.quote_rmse,
                    r.elapsed,
                    r.final_local_energy_change,
                ),
                ",",
            ),
        )
    end
end

println("\n[Summary by bond dimension]")
for χ in (32, 64, 96)
    selected = filter(r -> r.χ == χ, rows)
    @printf(
        "chi=%2d runs=%d median_residual=%.4e range=[%.4e, %.4e] median_quote=%.4e\n",
        χ,
        length(selected),
        median(getproperty.(selected, :residual)),
        minimum(getproperty.(selected, :residual)),
        maximum(getproperty.(selected, :residual)),
        median(getproperty.(selected, :quote_rmse)),
    )
end
@printf "best reference: chi=%d seed=%d E=%.10f residual=%.4e\n" reference.χ reference.seed reference.E reference.residual

χ_values = getproperty.(rows, :χ)
residual_values = getproperty.(rows, :residual)
quote_values = max.(getproperty.(rows, :quote_rmse), 1e-12)

residual_plot = scatter(
    χ_values,
    residual_values;
    yscale=:log10,
    xlabel="Bond dimension",
    ylabel="Ritz residual",
    xticks=[32, 64, 96],
    label="Independent runs",
    markersize=6,
    grid=true,
)
median_χ = [32, 64, 96]
median_residual = [
    median([r.residual for r in rows if r.χ == χ])
    for χ in median_χ
]
plot!(
    residual_plot,
    median_χ,
    median_residual;
    linewidth=2,
    marker=:diamond,
    label="Median",
)

quote_plot = scatter(
    χ_values,
    quote_values;
    yscale=:log10,
    xlabel="Bond dimension",
    ylabel="Quote RMSE vs. best run",
    xticks=[32, 64, 96],
    label="Independent runs",
    markersize=6,
    grid=true,
)
median_quote = [
    median([max(r.quote_rmse, 1e-12) for r in rows if r.χ == χ])
    for χ in median_χ
]
plot!(
    quote_plot,
    median_χ,
    median_quote;
    linewidth=2,
    marker=:diamond,
    label="Median",
)

comparison_plot = plot(residual_plot, quote_plot; layout=(1, 2), size=(950, 400))
savefig(comparison_plot, joinpath(OUTPUT_DIR, "strong_coupling_multiseed.pdf"))
savefig(comparison_plot, joinpath(OUTPUT_DIR, "strong_coupling_multiseed.png"))

println("\nResults written to $OUTPUT_DIR")
