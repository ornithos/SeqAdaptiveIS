# SeqAdaptiveIS

Sequential adaptive mixture importance sampling (sequential AMIS). This is an implementation of AMIS (Cappé et al. 2008), which iteratively improves a proposal distribution via re-fitting to the (weighted) importance samples. It's implemented such that it's easy to use on a sequence of target distributions ($p\_1, \ldots, p\_T$); i.e. such that the proposal adapted at time $t$ can be used as the initial proposal at time $t+1$.

Note that I wrote the source for this a couple of years before this refactoring, and so if correctness vs the paper is critical for you, please check the source, as I don't remember if there was any deviation from the paper. FWIW it's unlikely that there is anything major different, but it may differ in some small details.

The syntax is as follows (dumped from the help file:)

```julia
    amis(log_f, pis, mus, covs::AbstractArray; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)

    amis(log_f, S, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)

    amis(log_f, S, logW, k::Int; nepochs=5, gmm_smps=1000, IS_tilt=1., terminate=0.75, debug=false)
```

(1) `log_f`: the target function (distribution). This should take a single
argument: a set of $n \times d$ samples, and return a vector of length $n$ of
the log of the unnormalized target function (distribution) for each sample.

(2) `pis`: $k$-length probability vector of probabilities of each cluster.

(3) `mus`: matrix of cluster means: each *row* is a mean vector.

(4) `covs`: batched covariance matrices stacked in Tensor d×d×k for $k$
clusters.

Instead of args (2-4), i.e. the parameters of the GMM, one can instead supply a
matrix `S` of $n$ initial samples ($n \times d$ matrix), and optionally a
$n$-length vector `logW` of log weights corresponding to each sample. The final
positional argument is then an integer $k$ specifying the number of clusters to
use in the approximating GMM.

Optional kwargs are `nepochs`: the number of iterations of iterated IS;
`gmm_smps`, the number of samples to perform on each iteration, `IS_tilt`, the
exponential tilt applied to the proposal distribution at each epoch, `terminate`
specifies the proportion of `gmm_smps` the effective sample size (Owen, §9) must
reach before early stopping. ~~Finally `debug` provides some (2D) plots at each
epoch which I used when developing~~ [REMOVED TO AVOID LOADING PYPLOT].

The AMIS procedure is an iterated importance sampling (IIS) technique. At each
epoch, samples are drawn from a proposal distribution, which here is a Gaussian
Mixture Model. Each sample is assigned an importance weight vs. the target, and
a new proposal distribution is fitted to these weighted samples. Since we use
an iterative EM method for this, the outer IIS iterations are called epochs.
This procedure is iterated `nepochs` times.
