import Base: pop!
import Base: push!

mutable struct Buffer
    data::Matrix{Float32}
    count::Vector{Int64}
    stored::Vector{Int64}
    vacant::Vector{Int64}
end

function Buffer(capacity::Int64, dim::Int64)
    return Buffer(
        Matrix{Float32}(undef, (dim, capacity)),
        Vector{Int64}(undef, capacity),
        Vector{Int64}(),
        Vector{Int64}(1:capacity),
    )
end

function push!(buffer::Buffer, x::Matrix{Float32}, c::Union{Nothing, Vector{Int64}} = nothing)
    @assert length(buffer.vacant) >= size(x, 2)
    vacant_idx = StatsBase.sample(1:length(buffer.vacant), size(x, 2), replace=false, ordered=true)
    push_idx = splice!(buffer.vacant, vacant_idx)
    append!(buffer.stored, push_idx)
    buffer.data[:, push_idx] = x
    if isnothing(c)
        c = zeros(Int64, size(x, 2))
    end
    buffer.count[push_idx] = c
end

function pop!(buffer::Buffer, n::Int64)::Tuple{Matrix{Float32}, Vector{Int64}}
    @assert length(buffer.stored) >= n
    stored_idx = StatsBase.sample(1:length(buffer.stored), n, replace=false, ordered=true)
    pop_idx = splice!(buffer.stored, stored_idx)
    append!(buffer.vacant, pop_idx)
    return buffer.data[:, pop_idx], buffer.count[pop_idx]
end
