module SeqAdaptiveIS

using StatsBase

# Struct for holding results of inference.
struct MCPosterior
    samples::AbstractMatrix
    logW::Vector
end

StatsBase.weights(P::MCPosterior) = softmax(P.logW)
# resample(P::MCPosterior, N::Int) = P.samples[rand(Categorical(weights(P)), N),:]
resample(P::MCPosterior, N::Int) = P.samples[multicategorical_kitagawa(weights(P), N), :]
Base.length(P::MCPosterior) = length(P.logW)
ess(P::MCPosterior) = eff_ss(P.logW)

# useful class for holding components of GMM (unlike native impl, cholesky is not calculated)
# Not especially useful outside of this package, so it is not exported by default.
struct GMMComp{T <: Real}
    pis::Vector{T}
    mus::AbstractMatrix{T}
    covs::AbstractArray{T}
end

# AMIS procedure (and GMM / wEM fitting)
include("inference.jl")

# Application to sequential problems
include("sequential.jl")

export amis, resample, ess, seq_amis

end # module
