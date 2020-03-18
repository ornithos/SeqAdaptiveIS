# Make custom function for entry to AMIS so as to overload based on previous distribution
_amis_entry(log_f, previous::Distribution, S, logW, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75) =
    amis(log_f, previous, k; nepochs=nepochs, gmm_smps=gmm_smps, IS_tilt=IS_tilt, terminate=terminate)

_amis_entry(log_f, prev::GMMComp, S, logW, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75) =
    amis(log_f, prev.pis, prev.mus, prev.covs, S, logW; nepochs=nepochs, gmm_smps=gmm_smps, IS_tilt=IS_tilt, terminate=terminate)

"""
     seq_amis(target_logfs, init_dist::Union{Distribution, Int}, k::Int; nepochs=4, IS_tilt=1.3f0, gmm_smps=2000,
              terminate=0.6, max_retry=3, verbose=true)

Perform AMIS on a sequence of target log distributions. The initial distribution is specified with `init_dist`
and this function will perform AMIS using the previously tuned proposal as the initial distribution for the
next target.

Arguments:
- `target_logfs`: an iterable (e.g. vector) where element ``i`` corresponds to the ``i``th log target
    distribution. The distribution need not be normalized, and may often be a log joint.
- `init_dist`: a `Distribution` or `Int`, where the Int specifies the dimension of a standard multivariate
    Gaussian distribution. This is the initial proposal distribution for the first AMIS call.
- `k`: specifies the number of components to use for the proposal GMM.

Optional kwargs.
These are the sam as AMIS, except:
- `max_retry`: AMIS will sometimes fail to provide a good proposal (diagnosed by small ESS of the sample).
    This gives the maximum number of times AMIS will be retried (with increasing tilt and number of epochs).
    See also `min_ess`.
- `min_ess`: The minimum ESS required for each target distribution.
- `verbose`: Whether to display information regarding ESS (and retries) for each target.

Return value:
A tuple containing:
- A vector of the samples and corresponding log weights for each target. Each element can use the utility
    functions: `ess(x)`, `weights(x)` and `resample(x, n)`.
- A vector of the GMM components used for the final proposal for each target.
"""
function seq_amis(target_logfs, init_dist::Union{Distribution, Int}, k::Int; nepochs=4, IS_tilt=1.3f0,
                  gmm_smps=2000, terminate=0.6, max_retry=3, verbose=true, min_ess=min(100, gmm_smps/2))

    t0 = time()
    t_begin = copy(t0)
    if init_dist isa Int
        d = init_dist
        c_dist = MvNormal(zeros(Float32, d), convert(Matrix{Float32}, Matrix(I, d, d)))
    else
        c_dist = init_dist
    end

    smp_out = []
    dist_out = []

    pis, mus, covs = nothing, nothing, nothing    # local scope -- may not need since refactoring.
    S, logW, W = nothing, nothing, nothing

    for (i, lj) in enumerate(target_logfs)

        # reset quantities (in case changed due to non-convergence at prev iter)
        c_IS_tilt, c_nepochs, c_ess = IS_tilt, nepochs, 0

        for j = 1:max_retry
            S, logW, pis, mus, covs = _amis_entry(lj, c_dist, S, W, k; nepochs=c_nepochs, gmm_smps=gmm_smps,
                                                     IS_tilt=c_IS_tilt, terminate=terminate)
            W = softmax(logW)
            c_ess = eff_ss(W)
            (c_ess > 100) && break

            # increase computation / spread of proposal when ESS is low and retry
            # ==> This could be much smarter -- I'm currently throwing away the previous computation.
            # (Although note that not infrequently the previous computation is ~ uninformative.)
            IS_tilt, nepochs = IS_tilt*1.3, nepochs + 1
            verbose && printfmtln("Retrying... (ess = {:.1f})", c_ess)
        end
        c_dist = GMMComp(pis, mus, covs)

        push!(smp_out, MCPosterior(S, logW))
        push!(dist_out, GMMComp(pis, mus, covs))
        verbose && printfmtln("Posterior {:d}, ess={:.2f}, cum. time taken = {:.2f}s ({:.2f})", i, c_ess,
            time()-t_begin, time() - t0); flush(stdout);
        t0 = time()
    end

    return smp_out, dist_out
end
