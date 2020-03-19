using Distributions, StatsBase, StatsFuns, Random
using LinearAlgebra, Clustering
using Formatting


eye(d) = Matrix(I, d, d)
eff_ss(W) = 1/sum((W./sum(W)).^2)
weight_perp(W) = let w=W/sum(W); -sum(w.* log.(w))/log(length(W)) end


# softmax which accepts kwargs. This is actually slower than `softmax(x')'`
# in many cases, but :shrugs:, I wrote this a while ago, and is not usually the
# bottleneck of the code.
function softmax2(logp; dims=2)
    p = exp.(logp .- maximum(logp, dims=dims))
    p ./= sum(p, dims=dims)
    return p
end

function logsumexprows(X::AbstractArray{T}) where {T<:Real}
    n = size(X,1)
    out = zeros(T, n)
    for i = 1:n
        out[i] = logsumexp(X[i,:])
    end
    return out
end


# Kitagawa resampling (as used e.g. in SMC). This is sometimes useful if one is
# using resampling within a sequential AMIS scheme.
"""
    multicategorical_kitagawa(p::Vector{T}, m::Int64)

Draw `m` samples from the Categorical distribution defined by the probability
vector `p`. (Note that computationally there is no advantage to enforcing that
`p` sums to one, and hence ``\\sum_i p_i`` may be any positive number; it will
be implicitly renormalized.) This may be considered a stratified multinomial
sample, where the sample space is pre-partitioned into `m` intervals, each of
which obtains one random draw. Nevertheless, the number of repetitions of each
particle does not follow a multinomial distribution hence my clunky name,
"multicategorical". This follows the description given in Hol et al. 2006
("On Resampling Algorithms for Particle Filters"), but of course following
Kitagawa, 1996.
"""
function multicategorical_kitagawa(p::Vector{T}, m::Int64) where T <: AbstractFloat
    n = length(p)
    x = zeros(Int, m)     # return
    interval = sum(p)/m   # stratified intervals on [0,1) * sum(p) [sum p = 1 normally]
    cs = p[1]             # cum probability (calc online)
    i = 1                 # category index
#     println(" cs     lb    r     smp")
    for mm in 1:m
        lb = (mm-1)*interval
        r = rand()*interval
        while cs < lb + r
#             println(format("{:.3f} {:.3f} {:.3f} {:.3f}", cs, lb, r, lb+r))
            i += 1
            if i == n
                x[mm:end] .= i
                return x
            end
            @inbounds cs += p[i]
        end
#         println(format("{:.3f} {:.3f} {:.3f} {:.3f}", cs, lb, r, lb+r))
#         println("Choose i = ", i)
        @inbounds x[mm] = i
    end
    return x
end


################################################################################
##                                                                            ##
##                  Gaussian Mixture Model Utils                              ##
##    --------------------------------------------------------------------    ##
##    Custom GMM fitting utils: this uses both conjugate priors over GMM      ##
##    components, and allows for weighted observations: both are reqd for     ##
##    use in AMIS. I wrote this a year or so ago when Julia utils were        ##
##    less mature; probably some of this can be pointed at standard libs.     ##
##                                                                            ##
################################################################################


function gmm_llh(X, weights, pis, mus, sigmas; disp=false)
    n, p = size(X)
    k = length(pis)
    thrsh_comp = 0.005
    inactive_ixs = pis[:] .< thrsh_comp

    P = zeros(n, k)
    for j = 1:k
        P[:,j] = log_gauss_llh(X, mus[j,:], sigmas[:,:,j],
            bypass=inactive_ixs[j]) .+ log(pis[j])
    end
    P .*= weights
    if disp
        display(P)
        display(logsumexprows(P))
    end
    return logsumexprows(P)
end

function gmm_prior_llh(pis, mus, sigmas, pi_prior, mu_prior, cov_prior)
    # Prior over GMM: Using conjugate distributions (Normal-Inverse-Wishart IIRC)
    d = size(cov_prior, 1)
    ν = pi_prior # alias
    k = length(pis)
    out = zeros(k)
    @views for j = 1:k
        out[j] += logpdf(MvNormal(mu_prior[j,:], sigmas[:,:,j]/ν[j]), mus[j,:])
        out[j] += -(ν[j] + d + 1)*logdet(sigmas[:,:,j])/2
        out[j] += -ν[j]*sum(diag(cov_prior[:,:,j]*inv(sigmas[:,:,j])))/2
        out[j] += (ν[j] - 1)*log(pis[j])
    end
    return sum(out)
end


"""
    log_gauss_llh(X, mu, sigma; bypass=false)

loglikelihood of multivariate Gaussian. Probably no need for this anymore; it
was a bit faster than the built-in, but I think these have largely caught up.
The major point of this function is to have a `bypass`, which ignores the
entire calculation when convenient. Probably could be dealt with elsewhere,
but it's easier at the moment not to switch up the code when it works.
"""
function log_gauss_llh(X, mu, sigma; bypass=false)
    if bypass
        return -ones(size(X, 1))*Inf
    else
        retval = try _log_gauss_llh(X, mu, sigma)
            catch e
                return -ones(size(X, 1))*Inf
            end
        return retval
    end
end

function _log_gauss_llh(X, mu, sigma)
    d = size(X,2)
#     invUT = Matrix(cholesky(inv(sigma)).U)
    invUT = inv(cholesky(sigma).L)
    Z = (X .- mu')*invUT'
    exponent = -0.5*sum(Z.^2, dims=2)
    lognormconst = -d*log(2*pi)/2 -0.5*logdet(sigma) #.+ sum(log.(diag(invUT)))
    return exponent .+ lognormconst
end

"""
    gmm_custom(X, weights, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true)

Fitting a Gaussian Mixture Model (GMM) via MAP EM with *weighted* observations, and
priors over each of the parameters ``\\mu_j, \\Sigma_j, \\pi_j``, ``j \\in 1,\\ldots,k``.
The priors can be useful (i) to avoid exploding likelihood issues, and (ii) to provide
faster convergence given previous computation.
"""
function gmm_custom(X, weights, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true)
    n, p = size(X)
    k = length(pi_prior)
    @assert size(weights) == (n,)
    @assert size(mu_prior) == (k, p)
    @assert size(cov_prior) == (p, p, k)
    pis = pi_prior/sum(pi_prior)
    mus = copy(mu_prior)
    sigmas = copy(cov_prior)

    weights = weights / mean(weights)    # diff. to Cappé et al. due to prior

    thrsh_comp = 0.005
    inactive_ixs = pi_prior[:] .< thrsh_comp
    pi_prior = copy(pi_prior)
    Ns = zeros(6)

    for i in range(1, stop=max_iter)
        # E-step
        rs = reduce(hcat, map(j -> log_gauss_llh(X, mus[j,:], sigmas[:,:,j], bypass=inactive_ixs[j]), 1:k))
        try
            rs .+= log.(pis)[:]'
            catch e
            display(rs)
            display(log.(pis))
            rethrow(e)
        end

        rs = softmax2(rs, dims=2)
        # reweight according to importance weights (see Adaptive IS in General Mix. Cappé et al. 2008)
        rs .*= weights

        # M-step
        Ns = sum(rs, dims=1)
        inactive_ixs = Ns[:] .< 1
        active_ixs = .!inactive_ixs
        if any(inactive_ixs)
            pis[inactive_ixs] .= 0.0
            pi_prior[inactive_ixs] .= 0.0
        end
        pis = Ns[:] + pi_prior[:]

        pis /= sum(pis)

        _mus = reduce(vcat, map(j -> sum(X .* rs[:,j], dims=1) .+ pi_prior[j]*mu_prior[j,:]', findall(active_ixs)))
        _mus ./= vec(Ns[active_ixs] + pi_prior[active_ixs])
        mus[active_ixs,:] = _mus

        @views for j in findall(active_ixs)
            Δx = X .- mus[j, :]'
            Δμ = (mus[j,:] - mu_prior[j,:])'
            sigmas[:,:,j] = (Δx.*rs[:,j])'Δx + pi_prior[j]*(Δμ'Δμ + cov_prior[:,:,j])
            sigmas[:,:,j] ./= (Ns[j] + pi_prior[j] + p + 2)     # normalizing terms from Wishart prior
            sigmas[:,:,j] = (sigmas[:,:,j] + sigmas[:,:,j]')/2 + eye(p)*1e-6   # hack: prevent collapse
        end

    end

    return pis, mus, sigmas
end

function sample_from_gmm(n, pis, mus, covs; shuffle=true)
    k, p = size(mus)
    Ns = rand(Multinomial(n, pis[:]))
    active_ixs = findall(Ns[:] .>= 1)

    ixs = hcat(vcat(1, 1 .+ cumsum(Ns[1:end-1], dims=1)), cumsum(Ns, dims=1))
    out = zeros(n, p)
    for j=active_ixs
        out[ixs[j,1]:ixs[j,2],:] = rand(MvNormal(mus[j,:], covs[:,:,j]), Ns[j])'
    end
    if shuffle
        out = out[randperm(n),:]
    end
    return out
end


"""
    GMM_IS(n, pis, mus, covs, log_f)

GMM importance sampling. Sample `n` values from the GMM defined by the vectors
of parameters, `pis`, `mus`, `covs`, and return log importance weights relative
to the target distribution. The (unnormalized) target distribution is defined
by the log pdf `log_f`.
"""
function GMM_IS(n, pis, mus, covs, log_f)
    S = sample_from_gmm(n, pis, mus, covs, shuffle=false)
    logW = log_f(S) - gmm_llh(S, 1., pis, mus, covs, disp=false);
    return S, logW;
end


################################################################################
##                                                                            ##
##                  Adaptive Mixture Importance Sampling                      ##
##    --------------------------------------------------------------------    ##
##    Essentially following Cappé et al. 2008. This is the main (only)        ##
##    inference procedure in this file, which we use extensively for MTDS.    ##
##                                                                            ##
################################################################################


function amis(log_f, p::MvNormal, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)
    S = rand(p, gmm_smps)'
    logW = zeros(eltype(S), gmm_smps)
    amis(log_f, S, logW, k; nepochs=nepochs, gmm_smps=gmm_smps, IS_tilt=IS_tilt, terminate=terminate, debug=debug)
end


function amis(log_f, S, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)
    logW = zeros(size(S, 1))
    amis(log_f, S, logW, k; nepochs=nepochs, gmm_smps=gmm_smps, IS_tilt=IS_tilt, terminate=terminate, debug=debug)
end

function amis(log_f, S, logW, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)
    @assert !debug "debug facilities removed to reduce loading time (plots etc.)"
    n, p = size(S)

    W = softmax2(logW, dims=1)
    km = kmeans(copy(S'), k, weights=W)

    cmus = zeros(k, p)
    ccovs = zeros(p, p,k)
    for i in range(1, stop=k)
        ixs = findall(x -> isequal(x,i), km.assignments)
        cX = S[ixs, :]; cw = ProbabilityWeights(W[ixs])
        cmus[i,:] = cX' * cw/cw.sum
        ccovs[:,:,i] = StatsBase.cov(cX, cw, corrected=true)
    end

    cpis = zeros(k)
    cnts = countmap(km.assignments)

    for i in 1:k
        try
            cpis[i] = cnts[i]
        catch e
            @warn format("Error: {:s}. Commonly happens when cluster ({:d}) has no assigned points.", e.msg, i)
        end
    end
    cpis /= sum(cpis)
    # if debug
    #     f, axs = PyPlot.subplots(5,3, figsize=(8,12))
    #     plot_is_vs_target(S[:,1:2], W, ax=axs[1,1], c_num=7)
    #     for i = 1:k
    #         axs[1,2].plot(columns(AxPlot.utils.gaussian_2D_level_curve_pts(cmus[i,1:2], ccovs[1:2,1:2,i]))...);
    #     end
    # end

    amis(log_f, cpis, cmus, ccovs; nepochs=nepochs, gmm_smps=gmm_smps, IS_tilt=IS_tilt, terminate=terminate, debug=debug)
end


"""
    amis(log_f, pis, mus, covs::AbstractArray; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)

    amis(log_f, S, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)

    amis(log_f, S, logW, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)

Adaptive mixture importance sampling of Cappé et al. 2008. I wrote this a couple
of years before the docstring, so please check the source vs. the paper to
verify correctness; I may have made a few changes, but I don't recall now.
The arguments are:

(1) `log_f`: the target function (distribution). This should take a single
argument: a set of ``n \\times d`` samples, and return a vector of length ``n`` of
the log of the unnormalized target function (distribution) for each sample.

(2) `pis`: ``k``-length probability vector of probabilities of each cluster.

(3) `mus`: matrix of cluster means: each *row* is a mean vector.

(4) `covs`: batched covariance matrices stacked in Tensor d×d×k for `k`
clusters.

Instead of args (2-4), i.e. the parameters of the GMM, one can instead supply a
matrix `S` of ``n`` initial samples (``n \\times d`` matrix), and optionally a
``n``-length vector `logW` of log weights corresponding to each sample. The final
positional argument is then an integer `k` specifying the number of clusters to
use in the approximating GMM.

Optional kwargs are `nepochs`: the number of iterations of iterated IS;
`gmm_smps`, the number of samples to perform on each iteration, `IS_tilt`, the
exponential tilt applied to the proposal distribution at each epoch, `terminate`
specifies the proportion of `gmm_smps` the effective sample size (Owen, §9) must
reach before early stopping. Finally `debug` provides some (2D) plots at each
epoch which I used when developing.

The AMIS procedure is an iterated importance sampling (IIS) technique. At each
epoch, samples are drawn from a proposal distribution, which here is a Gaussian
Mixture Model. Each sample is assigned an importance weight vs. the target, and
a new proposal distribution is fitted to these weighted samples. Since we use
an iterative EM method for this, the outer IIS iterations are called epochs.
This procedure is iterated `nepochs` times.
"""
function amis(log_f, pis, mus, covs::AbstractArray; nepochs=5, gmm_smps=1000, IS_tilt=1.,
    terminate=0.75, debug=false)
    @assert !debug "debug facilities removed to reduce loading time (plots etc.)"

    S = sample_from_gmm(gmm_smps, pis, mus, covs*IS_tilt, shuffle=false)
    W = ones(eltype(mus), gmm_smps) / gmm_smps
    amis(log_f, pis, mus, covs, S, W; nepochs=nepochs, gmm_smps=gmm_smps,
        IS_tilt=IS_tilt, terminate=terminate, debug=debug)
end

function amis(log_f, pis, mus, covs::AbstractArray, S::AbstractMatrix, W::AbstractVector;
    nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)
    @assert !debug "debug facilities removed to reduce loading time (plots etc.)"

    ν_S, ν_W, log_W = S, W, nothing
    for i = 1:nepochs
        ## #TODO: inefficient, since most 'entry points' have that the weighted
        ## sample (S,W) is taken from the current GMM. Hence the following two
        ## lines should be moved to the *end* of the loop body. I'll need to do
        ## some checks to make sure it doesn't break anything.
        pis, mus, covs = gmm_custom(ν_S, ν_W, pis, mus, covs; max_iter=3, tol=1e-3, verbose=false);
        ν_S = sample_from_gmm(gmm_smps, pis, mus, covs*IS_tilt, shuffle=false)

        log_W = log_f(ν_S) - gmm_llh(ν_S, 1, pis, mus, covs*IS_tilt);
        ν_W = softmax(vec(log_W));

        (i == nepochs) || (eff_ss(ν_W) >= terminate * gmm_smps) && break
    end
    return ν_S, log_W, pis, mus, covs
end


# using PyPlot, AxPlot
# function plot_is_vs_target(S, W; ax=Nothing, c_num=1, cm="tab10", kwargs...)
# 	@assert cm == "tab10" || cm == "Blues"
# 	col_matrix = cm == "tab10" ? cols_tab10 : cols_Blues
# 	rgba_colors = repeat(col_matrix[c_num:c_num,:], size(S, 1), 1)
#     rgba_colors[:, 4] = W/maximum(W)   # alpha
# #     print(rgba_colors)
#     ax = ax==Nothing ? gca() : ax
#     # display(S)
# #     plot_level_curves_all(mus, UTs; ax=ax)
#     ax[:scatter](splat(S)..., c=rgba_colors, kwargs...)
# end
