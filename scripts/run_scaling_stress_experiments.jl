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

"""
Construct a factor model for which
`opnorm(BB') / minimum(Δ) == coupling_ratio`.
This makes factor strength comparable across different `N` and `K`.
"""
function controlled_factor_model(
    N::Int,
    K::Int;
    Q::Int=2,
    coupling_ratio::Float64=1.0,
    seed::Int=1,
)
    rng = MersenneTwister(seed)
    Δ = 0.6 .+ 0.3 .* rand(rng, N)
    η = 0.9 .+ 0.2 .* rand(rng, N)
    Braw = randn(rng, N, K)
    B = sqrt(coupling_ratio * minimum(Δ)) .* Braw ./ opnorm(Braw)
    Σ = Diagonal(Δ) + B * B'
    model = MarketMakingModel(N, fill(Q, N), Matrix(Σ), zeros(N), 1.0, 1.1, η)
    cores = build_mpo_cores(model, Δ, B)
    actual_ratio = opnorm(B)^2 / minimum(Δ)
    return model, Δ, B, cores, actual_ratio
end

function sampled_edges(N::Int, count::Int; seed::Int)
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

function sampled_quote_rmse(mps, reference, edges, Qs)
    squared_error = 0.0
    for (q, qn) in edges
        difference =
            mps_log_ratio(mps, q, qn, Qs) -
            mps_log_ratio(reference, q, qn, Qs)
        squared_error += difference^2
    end
    return sqrt(squared_error / length(edges))
end

cases = [
    (name="N20", N=20, K=3, Q=2, ratio=1.0, seed=120),
    (name="N50", N=50, K=3, Q=2, ratio=1.0, seed=150),
    (name="N100", N=100, K=3, Q=2, ratio=1.0, seed=200),
    (name="weak", N=50, K=3, Q=2, ratio=0.1, seed=150),
    (name="strong", N=50, K=3, Q=2, ratio=10.0, seed=150),
    (name="K1", N=50, K=1, Q=2, ratio=1.0, seed=150),
    (name="K5", N=50, K=5, Q=2, ratio=1.0, seed=150),
    (name="d11", N=50, K=3, Q=5, ratio=1.0, seed=150),
]
χs = [4, 8, 16, 32]
rows = NamedTuple[]

println("[Controlled scaling and stress sweep]")
println("Residual is scaled by max(abs(E), 1); quote RMSE uses chi=32 as reference.")

for (case_index, case) in enumerate(cases)
    model, Δ, B, cores, actual_ratio = controlled_factor_model(
        case.N,
        case.K;
        Q=case.Q,
        coupling_ratio=case.ratio,
        seed=case.seed,
    )
    edges = sampled_edges(case.N, 300; seed=10_000 + case_index)
    solutions = NamedTuple[]
    n_sweeps = case.N <= 20 ? 8 : 6

    for χ in χs
        elapsed = @elapsed begin
            E, mps, _ = dmrg_ground_state(
                cores;
                maxdim=χ,
                n_sweeps=n_sweeps,
                cutoff=1e-13,
                rng=MersenneTwister(20_000 + 100 * case_index + χ),
            )
        end
        residual = dmrg_residual(mps, cores, E)
        scaled_residual = residual / max(abs(E), 1.0)
        push!(solutions, (; χ, E, mps, residual, scaled_residual, elapsed))
    end

    reference = solutions[end].mps
    for solution in solutions
        quote_rmse =
            solution.χ == χs[end] ? 0.0 :
            sampled_quote_rmse(solution.mps, reference, edges, model.Qs)
        row = (;
            case=case.name,
            N=case.N,
            K=case.K,
            d=2 * case.Q + 1,
            target_ratio=case.ratio,
            actual_ratio,
            χ=solution.χ,
            E=solution.E,
            residual=solution.residual,
            scaled_residual=solution.scaled_residual,
            quote_rmse,
            elapsed=solution.elapsed,
        )
        push!(rows, row)
        @printf(
            "%-6s N=%3d K=%d d=%2d ratio=%4.1f chi=%2d relres=%9.2e quote=%9.2e time=%6.1fs\n",
            row.case,
            row.N,
            row.K,
            row.d,
            row.actual_ratio,
            row.χ,
            row.scaled_residual,
            row.quote_rmse,
            row.elapsed,
        )
    end
end

open(joinpath(OUTPUT_DIR, "scaling_stress.csv"), "w") do io
    println(
        io,
        "case,N,K,d,target_ratio,actual_ratio,chi,E,residual,scaled_residual,quote_rmse,elapsed_seconds",
    )
    for r in rows
        println(
            io,
            join(
                (
                    r.case,
                    r.N,
                    r.K,
                    r.d,
                    r.target_ratio,
                    r.actual_ratio,
                    r.χ,
                    r.E,
                    r.residual,
                    r.scaled_residual,
                    r.quote_rmse,
                    r.elapsed,
                ),
                ",",
            ),
        )
    end
end

function values_for(case_name::String, field::Symbol; omit_reference::Bool=false)
    selected = filter(r -> r.case == case_name, rows)
    omit_reference && (selected = filter(r -> r.χ != maximum(χs), selected))
    return getproperty.(selected, :χ), getproperty.(selected, field)
end

residual_plot = plot(
    yscale=:log10,
    xlabel="Bond dimension",
    ylabel="Ritz residual",
    xticks=χs,
    grid=true,
)
quote_plot = plot(
    yscale=:log10,
    xlabel="Bond dimension",
    ylabel="Sampled quote RMSE",
    xticks=χs[1:end-1],
    grid=true,
)
for (name, label, marker) in (
    ("N20", "N=20", :circle),
    ("N50", "N=50", :square),
    ("N100", "N=100", :diamond),
)
    xres, yres = values_for(name, :residual)
    plot!(residual_plot, xres, yres; marker, linewidth=2, label)
    xquote, yquote = values_for(name, :quote_rmse; omit_reference=true)
    plot!(quote_plot, xquote, yquote; marker, linewidth=2, label)
end
scaling_plot = plot(residual_plot, quote_plot; layout=(1, 2), size=(950, 400))
savefig(scaling_plot, joinpath(OUTPUT_DIR, "scaling_with_assets.pdf"))
savefig(scaling_plot, joinpath(OUTPUT_DIR, "scaling_with_assets.png"))

stress_names = ["weak", "N50", "strong", "K1", "K5", "d11"]
stress_labels = ["ratio 0.1", "ratio 1", "ratio 10", "K=1", "K=5", "d=11"]
stress_residual = Float64[]
stress_quote = Float64[]
for name in stress_names
    selected = filter(r -> r.case == name && r.χ == 8, rows)
    push!(stress_residual, only(selected).residual)
    push!(stress_quote, only(selected).quote_rmse)
end

stress_residual_plot = bar(
    stress_labels,
    stress_residual;
    yscale=:log10,
    ylabel="Ritz residual",
    legend=false,
    xrotation=25,
    grid=:y,
)
stress_quote_plot = bar(
    stress_labels,
    stress_quote;
    yscale=:log10,
    ylabel="Quote RMSE vs. chi=32",
    legend=false,
    xrotation=25,
    grid=:y,
)
stress_plot = plot(stress_residual_plot, stress_quote_plot; layout=(1, 2), size=(950, 420))
savefig(stress_plot, joinpath(OUTPUT_DIR, "scaling_stress_chi8.pdf"))
savefig(stress_plot, joinpath(OUTPUT_DIR, "scaling_stress_chi8.png"))

println("\nResults written to $OUTPUT_DIR")
