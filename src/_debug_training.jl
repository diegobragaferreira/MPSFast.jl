#!/usr/bin/env julia
# Run: julia --project=. src/_debug_training.jl
using MPSFast
using MPSFast.Encoders
using Random

println("Julia version: ", VERSION)
Random.seed!(42)

m   = 4   # d = 16
enc = BasisEncoder(m)

# tiny synthetic paths
Nd, M_steps = 100, 20
paths = 90 .+ 25 .* rand(Float32, Nd, M_steps)
fit_grid!(enc, paths)
xi = encode_paths(enc, paths)
println("xi: ", size(xi), "  max=", maximum(xi))

Ml  = chain_length(enc, M_steps)
d   = site_dim(enc)
Dmax = 20

mps = init_mps(Ml, d, Dmax; rng = MersenneTwister(1))
println("mps sizes: ", [size(t) for t in mps])

# Check bonds are all positive
for j in 1:length(mps)
    for k in (1, 2, 3)
        @assert size(mps[j], k) > 0 "mps[$j] dim $k = $(size(mps[j],k))"
    end
end
println("All MPS bonds are positive ✓")

# One manual update_pair! call to catch the crash
ws  = MPSFast.TrainWorkspace(Float32, Nd, d, Dmax)
Lenv, Renv = MPSFast.norm_environments(mps)
adam = MPSFast.AdamDict{Float32}()
Lv   = ones(Float32, Nd, 1)
j = 1
Dl = size(mps[j], 1); d_ = size(mps[j], 2); Dr = size(mps[j+1], 3)
println("Bond j=$j: Dl=$Dl, d=$d_, Dr=$Dr")
println("B_hat would be shape ($(Dl*d_), $(d_*Dr))")
println("keep_target = $(min(Dmax, min(Dl*d_, d_*Dr)))")

println("\nCalling update_pair! ...")
MPSFast.update_pair!(ws, mps, xi, j, 5e-4, Dmax, 1f-5, Lenv, Renv, adam;
    Lv_carry = Lv, epoch = 1, sweep = :forward, d_phys = d)
println("update_pair! succeeded ✓")

println("\nRunning train_mps! for 1 epoch ...")
train_mps!(mps, xi, 1, 5e-4, Dmax, 1e-5; verbose = true, nll_samples = 50)
println("Training succeeded ✓")
