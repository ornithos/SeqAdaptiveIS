println("Testing...")
using Test
using SeqAdaptiveIS
using Distributions, StatsFuns

# These are fairly rudimentary 2D tests with easy targets: I guess their primary
# purpose is to ensure that nothing has *broken* the procedures, but is not
# sufficiently strong enough yet to tell if it breaks in difficult or edge
# cases.

# ======================== AMIS inference tests ================================
p_x = MvNormal(zeros(2), Float64[1 0; 0 1])
test_lpdf64_v0 = (x-> logpdf(MvNormal(zeros(2), Float64[1 0; 0 1]), x'), p_x, zeros(2), Float64[1 0; 0 1])
test_lpdf64_v1 = (x-> logpdf(MvNormal(ones(2), Float64[1 0; 0 1]), x'), p_x, ones(2), Float64[1 0; 0 1])

function test_amis(test_case, k::Int)
    log_target, prior, μ, Σ = test_case
    S, logW, pis, mus, covs = amis(log_target, prior, k)
    W = softmax(logW)
    μ̂ = vec(sum(S .* W, dims=1))
    Σ̂ = (S .- μ̂')' * ((S .- μ̂') .* W)
    @test maximum(abs.(μ - μ̂)) < 0.1
    @test maximum(vec(Σ - Σ̂)) < 0.3
end


test_amis(test_lpdf64_v0, 1)
test_amis(test_lpdf64_v0, 2)
test_amis(test_lpdf64_v0, 5)
test_amis(test_lpdf64_v1, 1)
test_amis(test_lpdf64_v1, 2)
test_amis(test_lpdf64_v1, 5)

# ============================= Seq AMIS test ==================================

n_targets = 8
true_pars = [(zeros(2) .+ i, Float64[1 0; 0 1] ./ i) for i in 1:n_targets];
seq_targets = [x->logpdf(MvNormal(μ, Σ), x') for (μ, Σ) in true_pars]

seq_smps, seq_gmms = seq_amis(seq_targets, 2, 3)
print("Sequential targets complete: ")
for i in 1:n_targets
    μ, Σ = true_pars[i]
    S = resample(seq_smps[i], 1000)
    μ̂ = vec(mean(S, dims=1))
    Σ̂ = cov(S, dims=1)
    print("|")
    @test maximum(abs.(μ - μ̂)) < 0.1
    @test maximum(vec(Σ - Σ̂)) < 0.3
end
println()
