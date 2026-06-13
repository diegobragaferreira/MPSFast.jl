# io.jl — JLD2 checkpointing for MPS bundles.

"""
    save_mps_bundle(path, mps, nll_hist, epoch, meta; bond_log = nothing)

Persist the current MPS, NLL history, epoch counter, metadata dictionary,
and an optional bond-spectrum log to a JLD2 file at `path`.
"""
function save_mps_bundle(
    path::AbstractString, mps, nll_hist, epoch::Int,
    meta::Dict{String,Any};
    bond_log = nothing,
)
    if bond_log === nothing
        jldsave(path; mps = mps, nll_hist = collect(nll_hist),
                      epoch = epoch, meta = meta)
    else
        jldsave(path; mps = mps, nll_hist = collect(nll_hist),
                      epoch = epoch, meta = meta,
                      bond_log = collect(bond_log))
    end
end

"""
    load_mps_bundle(path) -> (mps, nll_hist, epoch, meta)

Load an MPS bundle saved with `save_mps_bundle`.
"""
function load_mps_bundle(path::AbstractString)
    d    = load(path)
    meta = get(d, "meta", Dict{String,Any}())
    return d["mps"], d["nll_hist"], d["epoch"], meta
end

"""
    load_bond_log(path) -> Vector or nothing

Return the `bond_log` field if it was saved in the bundle, otherwise `nothing`.
"""
function load_bond_log(path::AbstractString)
    d = load(path)
    return get(d, "bond_log", nothing)
end
