"""
Phase 2 / DeltaOFV: is the recovered likelihood actually usable for model selection?
=====================================================================================

THE QUESTION THIS SCRIPT ANSWERS:

    is_marginal_loglik (nlmevi_core) gives an importance-sampling estimate of
    the marginal log-likelihood, on NONMEM's OFV scale. That's necessary but
    not sufficient for claiming "you can run a likelihood-ratio test with
    this." This script checks that claim directly: simulate data under a
    known NULL, fit nested models, and see whether the empirical distribution
    of dOFV = OFV_reduced - OFV_full matches its theoretical reference.

    If it doesn't, the likelihood recovery is not yet fit for model
    selection, regardless of how good the point estimates look.

THE STATISTICAL SUBTLETY THIS SCRIPT DOES NOT SKIP:

    The nested comparison here is FULL model (3 random effects: eta_CL,
    eta_V, eta_Ka) vs REDUCED model (2 random effects: eta_CL, eta_V; ka has
    NO between-subject variability). This is a test of whether a variance
    component (om_Ka) is zero.

    A naive analyst reaches for dOFV ~ chi-square(df=1) here. That is WRONG.
    Testing a variance against the boundary of its parameter space (Var >= 0)
    is a non-regular problem: under the null, the MLE of the variance is at
    the boundary (0) roughly half the time, which point-masses half the null
    distribution's probability at dOFV = 0. The correct asymptotic reference
    (Self & Liang 1987, Case 5) is a 50:50 MIXTURE of a point mass at 0 and
    chi-square(df=1), NOT plain chi-square(df=1). This is the same subtlety
    that applies when testing OMEGA=0 in NONMEM. Calibrating against plain
    chi-square(1) here would make the test falsely conservative and the
    reported "validation" would be wrong even if it looked reassuring.

    This script calibrates against the correct mixture reference and reports
    both: (a) the fraction of null replicates landing at the boundary
    (should be ~50%), and (b) whether the strictly-positive dOFV values
    among the rest follow chi-square(1) (KS test + QQ data).

DESIGN
    - Simulate N replicates under the REDUCED model (true om_Ka = 0 --
      i.e., ka has zero between-subject variability, generated via
      nlmevi_core.simulate with TrueParams.om_ka=0.0).
    - Fit BOTH models to each replicate:
        reduced: OneCmtOralNoKaRE (n_eta=2, 6 population params)
        full:    OneCmtOral        (n_eta=3, 7 population params)
    - dOFV = -2*(ll_reduced - ll_full) = OFV_reduced - OFV_full
    - Compare the empirical distribution of dOFV to the Self-Liang mixture.

REQUIRES  nlmevi_core.py in the same directory.
"""

# %% ---------------------------------------------------------------- imports
import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlmevi_core import (
    TrueParams, OneCmtOral, simulate, to_tensors, fit_model, conc_analytic,
)

torch.set_default_dtype(torch.float64)


# %% ------------------------------------------------- the reduced model
class OneCmtOralNoKaRE:
    """
    Identical to OneCmtOral except ka has NO random effect: only eta_CL and
    eta_V are estimated (n_eta=2). ka is a pure fixed effect.

    theta (log scale, 6 params instead of 7):
        [log CL, log V, log ka, log om_CL, log om_V, log sigma]

    This is the REDUCED model in the nested comparison. Note it still
    ACCEPTS a 3-column eta tensor's worth of noise from upstream samplers if
    ever reused generically -- but here n_eta=2 so the posterior samplers
    (make_posterior) correctly allocate only 2 random effects, keeping the
    parameter count difference between reduced (6) and full (7) exactly 1,
    which is what makes df=1 the right reference.
    """
    n_eta = 2
    n_theta = 6

    def __init__(self, dose):
        self.dose = dose

    @staticmethod
    def unpack(theta):
        tv = theta[:3]              # log CL, log V, log ka (ka has no eta)
        om = torch.exp(theta[3:5])  # om_CL, om_V only
        sigma = torch.exp(theta[5])
        return tv, om, sigma

    def log_prior(self, eta, theta):
        _, om, _ = self.unpack(theta)
        return (-0.5 * (eta / om) ** 2 - torch.log(om) - 0.5 * math.log(2 * math.pi)).sum(-1)

    def predict(self, eta, theta, t):
        tv, _, _ = self.unpack(theta)
        CL = torch.exp(tv[0] + eta[..., 0:1])
        V = torch.exp(tv[1] + eta[..., 1:2])
        ka = torch.exp(tv[2])   # no eta -- identical for every subject/sample
        return conc_analytic(t, CL, V, ka, self.dose)

    def log_lik(self, logy, eta, theta, t, mask):
        _, _, sigma = self.unpack(theta)
        pred = self.predict(eta, theta, t).clamp_min(1e-12)
        resid = logy - torch.log(pred)
        ll = -0.5 * (resid / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return (ll * mask).sum(-1)

    def log_joint(self, logy, eta, theta, t, mask):
        return self.log_prior(eta, theta) + self.log_lik(logy, eta, theta, t, mask)


# %% ------------------------------------------------------------ experiment
def _fit_one_replicate(rep, seed, n_subj, K, max_steps, eval_k=4000, posterior="free"):
    """
    Does the full reduced+full fit for ONE replicate. Module-level and only
    takes plain picklable arguments (no torch tensors, no model objects) so
    it can be dispatched to a separate process -- each worker rebuilds
    everything from scratch, since nothing about a fit depends on another
    replicate's state.

    eval_k passed explicitly (not via monkey-patched defaults) specifically
    because it must reach worker processes correctly: each worker
    re-imports nlmevi_core fresh (spawn), so any patch applied to a module
    default in the parent process would silently NOT propagate to workers.
    A plain function argument always propagates correctly regardless of
    process boundaries.

    posterior="free" (default) vs "amortized" tests a specific hypothesis
    about WHY dOFV is miscalibrated: the free posterior gives the full
    model (n_eta=3) one extra free (mu, log_s) pair PER SUBJECT relative to
    the reduced model (n_eta=2) -- e.g. 120 extra parameter-pairs at
    N=120, not just the single population-level Omega_ka being tested. The
    Self-Liang boundary-mixture reference assumes exact MLE and a
    difference of exactly ONE degree of freedom; this much extra
    per-subject capacity could let VI fit subject-specific noise better
    than exact MLE would, inflating dOFV beyond what theory predicts.
    Amortized uses a SHARED encoder across subjects -- the full model then
    only adds one extra output dimension to a shared network, not N extra
    free parameter-pairs. If amortized calibrates meaningfully better than
    free, that's direct evidence for this explanation.

    torch.set_num_threads(1) here is not optional: PyTorch defaults to
    using multiple CPU threads internally for its own ops. Combine that
    with multiple WORKER PROCESSES each doing the same thing and you get
    oversubscription -- more threads fighting over the same cores than the
    machine has, which makes total throughput WORSE than running serially.
    Pinning each worker to 1 thread and letting process-level parallelism
    provide the actual parallelism is what makes this a net speedup.
    """
    torch.set_num_threads(1)
    tp = TrueParams()
    tp.om_ka = 0.0
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]

    theta_reduced_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                          math.log(0.3), math.log(0.3), math.log(0.3)]
    theta_full_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                       math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]
    model_reduced = OneCmtOralNoKaRE(tp.dose)
    model_full = OneCmtOral(tp.dose)

    data = simulate(tp, n_subj, times, seed=seed)   # ONE dataset for both fits

    t0 = time.time()
    theta_r, _, ll_red, _, _, n_run_r, conv_r, _ = fit_model(
        model_reduced, data, posterior, "gaussian", K, theta_reduced_init,
        n_eta=2, max_steps=max_steps, eval_k=eval_k,
    )
    theta_f, _, ll_full, _, _, n_run_f, conv_f, _ = fit_model(
        model_full, data, posterior, "gaussian", K, theta_full_init,
        n_eta=3, max_steps=max_steps, eval_k=eval_k,
    )
    dofv = -2 * (ll_red - ll_full)

    # Full parameter sets from both fits, not just dOFV -- specifically
    # added to test whether SIGMA (residual error) diverges between the
    # reduced and full fits on the same data. This is the live, untested
    # hypothesis for the residual calibration gap that persisted even
    # after switching to an amortized posterior (which fixed the
    # boundary-mass component but left a smaller, unexplained deviation in
    # the shape of the positive-dOFV tail, confirmed at both n=100 and
    # n=300, not explained by non-convergence). If sigma_full and
    # sigma_reduced differ substantially on the same data, that's another
    # unaccounted source of asymmetry beyond the single Omega_ka parameter
    # nominally being tested -- a second, independent lever the classical
    # boundary-mixture reference doesn't account for, on top of the
    # per-subject-parameter effect already identified.
    # OneCmtOralNoKaRE.theta layout: [logCL, logV, logka, logom_CL, logom_V, logsigma]
    # OneCmtOral.theta layout:       [logCL, logV, logka, logom_CL, logom_V, logom_ka, logsigma]
    return dict(
        replicate=rep, seed=seed, ll_reduced=ll_red, ll_full=ll_full,
        dofv=dofv, converged_reduced=conv_r, converged_full=conv_f,
        n_steps_reduced=n_run_r, n_steps_full=n_run_f,
        secs=time.time() - t0,
        CL_reduced=math.exp(theta_r[0]), V_reduced=math.exp(theta_r[1]),
        ka_reduced=math.exp(theta_r[2]), om_CL_reduced=math.exp(theta_r[3]),
        om_V_reduced=math.exp(theta_r[4]), sigma_reduced=math.exp(theta_r[5]),
        CL_full=math.exp(theta_f[0]), V_full=math.exp(theta_f[1]),
        ka_full=math.exp(theta_f[2]), om_CL_full=math.exp(theta_f[3]),
        om_V_full=math.exp(theta_f[4]), om_ka_full=math.exp(theta_f[5]),
        sigma_full=math.exp(theta_f[6]),
        sigma_diff=math.exp(theta_f[6]) - math.exp(theta_r[5]),
    )


def run_null_calibration(n_reps, n_subj, K, max_steps, out, seed0=5000,
                         n_workers=1, eval_k=4000, posterior="free"):
    """
    Simulates n_reps null datasets (true om_Ka = 0), fits reduced and full
    models to each, and returns the per-replicate dOFV plus both models'
    parameter estimates for diagnostic purposes.

    n_workers=1 (default): original sequential behavior, unchanged.
    n_workers>1: dispatches replicates across processes via
    ProcessPoolExecutor. Replicates are fully independent (each simulates
    its own data and fits both models from scratch), so this is
    embarrassingly parallel -- no correctness difference from the
    sequential version, only wall-clock time. Results are collected and
    then sorted back into replicate order, since processes complete out of
    order.
    """
    if n_workers <= 1:
        rows = []
        for rep in range(n_reps):
            seed = seed0 + rep
            row = _fit_one_replicate(rep, seed, n_subj, K, max_steps,
                                     eval_k=eval_k, posterior=posterior)
            rows.append(row)
            flag = "" if (row["converged_reduced"] and row["converged_full"]) else \
                "  *** CONVERGENCE ISSUE ***"
            print(f"  rep {rep:3d} | ll_reduced {row['ll_reduced']:8.2f} | "
                 f"ll_full {row['ll_full']:8.2f} | dOFV {row['dofv']:+7.3f} | "
                 f"{row['secs']:5.1f}s{flag}")
        return pd.DataFrame(rows)

    print(f"  Running {n_reps} replicates across {n_workers} worker processes...")
    rows = [None] * n_reps
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_fit_one_replicate, rep, seed0 + rep, n_subj, K,
                             max_steps, eval_k, posterior): rep
                  for rep in range(n_reps)}
        for fut in as_completed(futures):
            row = fut.result()
            rows[row["replicate"]] = row
            n_done += 1
            flag = "" if (row["converged_reduced"] and row["converged_full"]) else \
                "  *** CONVERGENCE ISSUE ***"
            print(f"  [{n_done:3d}/{n_reps}] rep {row['replicate']:3d} | "
                 f"ll_reduced {row['ll_reduced']:8.2f} | ll_full {row['ll_full']:8.2f} | "
                 f"dOFV {row['dofv']:+7.3f} | {row['secs']:5.1f}s{flag}")
    return pd.DataFrame(rows)


def calibrate(df, out):
    """
    Compares the empirical dOFV distribution to the Self-Liang (1987)
    boundary mixture: 0.5 * point-mass-at-0 + 0.5 * chi-square(df=1).

    Under a correctly-behaving likelihood:
      - roughly HALF of replicates should show dOFV <= ~0 (the reduced
        model's om_Ka estimate hit or near the boundary -- adding the extra
        random effect didn't help, as it shouldn't under the true null)
      - the other half, dOFV should follow chi-square(1)
    """
    dofv = df.dofv.values
    n = len(dofv)

    frac_at_boundary = np.mean(dofv <= 0.05)   # small tolerance for optimizer noise
    print("\n" + "=" * 72)
    print("CALIBRATION -- comparing empirical dOFV to the Self-Liang mixture")
    print("=" * 72)
    print(f"  N replicates: {n}")
    print(f"  Fraction with dOFV <= 0 (should be ~50% under the mixture, "
          f"NOT under plain chi-sq(1)): {100*frac_at_boundary:.1f}%")

    positive = dofv[dofv > 0.05]
    if len(positive) >= 10:
        # KS test: do the strictly-positive dOFV values follow chi-square(1)?
        ks_stat, ks_p = stats.kstest(positive, "chi2", args=(1,))
        print(f"  KS test of positive dOFV vs chi-square(1): "
              f"stat={ks_stat:.4f}, p={ks_p:.4f}  "
              f"{'PASS (fail to reject)' if ks_p > 0.05 else 'FAIL -- reference mismatch'}")
    else:
        print("  Too few positive-dOFV replicates for a KS test -- increase --reps")
        ks_p = None

    # Contrast: naive (WRONG) reference vs the correct Self-Liang mixture
    # reference, both at nominal alpha=0.05.
    #   - Naive analyst tests dOFV against chi-square(1)'s 95th percentile.
    #     Under the TRUE mixture null, P(dOFV > chi2.ppf(0.95,1)) =
    #     0.5 * P(chi2(1) > chi2.ppf(0.95,1)) = 0.5*0.05 = 2.5%, not 5% --
    #     the naive test is silently conservative (underpowered), not merely
    #     "a bit off."
    #   - Correct mixture test rejects at chi2.ppf(0.90,1) instead (Self &
    #     Liang 1987): P(dOFV > chi2.ppf(0.90,1)) = 0.5*P(chi2(1)>chi2.ppf(0.90,1))
    #     = 0.5*0.10 = 5%, exactly nominal.
    naive_cutoff = stats.chi2.ppf(0.95, df=1)
    correct_cutoff = stats.chi2.ppf(0.90, df=1)
    naive_reject_rate = np.mean(dofv > naive_cutoff)
    correct_reject_rate = np.mean(dofv > correct_cutoff)
    print(f"\n  Naive (WRONG) test -- reject if dOFV > chi2.ppf(0.95,1)={naive_cutoff:.3f}:")
    print(f"    empirical type-I error = {100*naive_reject_rate:.1f}%  "
          f"(true value under the mixture is 2.5%, not the intended 5% -- "
          f"this is why 'just use chi-square(1)' quietly under-powers the test)")
    print(f"  Correct (Self-Liang) test -- reject if dOFV > chi2.ppf(0.90,1)={correct_cutoff:.3f}:")
    print(f"    empirical type-I error = {100*correct_reject_rate:.1f}%  (target: 5%)")

    # Sigma-divergence hypothesis test: if the reduced and full fits'
    # residual error estimates diverge substantially on the same data,
    # that's an unaccounted asymmetry beyond the single tested parameter
    # (Omega_ka) -- a second potential explanation for the residual
    # calibration gap that persisted after switching to an amortized
    # posterior (which fixed the boundary-mass component but left the
    # positive-tail shape still deviating from chi-square(1)).
    if "sigma_diff" in df.columns:
        print("\n" + "-" * 72)
        print("SIGMA-DIVERGENCE CHECK -- does sigma_full - sigma_reduced explain")
        print("the residual gap? (untested hypothesis, now checkable)")
        print("-" * 72)
        sigma_diff = df.sigma_diff.values
        print(f"  sigma_diff (full - reduced): mean={sigma_diff.mean():+.4f}  "
              f"sd={sigma_diff.std():.4f}  "
              f"range=[{sigma_diff.min():+.4f}, {sigma_diff.max():+.4f}]")
        corr = np.corrcoef(sigma_diff, dofv)[0, 1]
        corr_abs = np.corrcoef(sigma_diff, np.abs(dofv))[0, 1]
        print(f"  corr(sigma_diff, dOFV)      = {corr:+.3f}")
        print(f"  corr(sigma_diff, |dOFV|)    = {corr_abs:+.3f}")
        # Compare sigma_diff for the most extreme-|dOFV| decile vs the rest --
        # a cleaner signal than a single correlation coefficient if the
        # relationship is nonlinear or concentrated in the tail.
        thresh = np.percentile(np.abs(dofv), 90)
        extreme = np.abs(dofv) >= thresh
        print(f"  mean |sigma_diff|, top-10% |dOFV| replicates (n={extreme.sum()}): "
              f"{np.abs(sigma_diff[extreme]).mean():.4f}")
        print(f"  mean |sigma_diff|, remaining 90% (n={(~extreme).sum()}): "
              f"{np.abs(sigma_diff[~extreme]).mean():.4f}")
        if abs(corr_abs) > 0.3:
            print("  -> Meaningful correlation: sigma divergence IS associated with")
            print("     dOFV magnitude. Worth pursuing as the explanation for the")
            print("     residual gap.")
        else:
            print("  -> Weak/no correlation: sigma divergence does NOT appear to")
            print("     explain the residual gap. Rule this hypothesis out and look")
            print("     elsewhere.")

    verdict_ok = abs(frac_at_boundary - 0.5) < 0.15 and (ks_p is None or ks_p > 0.01)
    print("\n" + "-" * 72)
    if verdict_ok:
        print("VERDICT: dOFV is well-calibrated against the correct boundary-mixture")
        print("reference. Likelihood-ratio tests using this likelihood recovery are")
        print("trustworthy for this class of comparison (testing a variance component).")
    else:
        print("VERDICT: dOFV does NOT match the expected reference. Do not use this")
        print("likelihood recovery for LRTs yet -- something in the marginal-likelihood")
        print("estimate (K, convergence, or the IS proposal itself) needs attention")
        print("before it can be trusted for model selection claims.")

    # Figure: empirical dOFV histogram vs the mixture density
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dofv, bins=30, density=True, alpha=0.6, label="empirical dOFV")
    xx = np.linspace(0.01, max(dofv.max(), 8), 300)
    ax.plot(xx, 0.5 * stats.chi2.pdf(xx, df=1), "r-", lw=2,
           label="0.5 x chi-sq(1) density (positive half of the mixture)")
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("dOFV = OFV_reduced - OFV_full")
    ax.set_ylabel("density")
    ax.set_title(f"dOFV calibration under H0 (true om_Ka=0)\n"
                 f"{100*frac_at_boundary:.0f}% at boundary (target ~50%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/phase2_deltaofv_calibration.png", dpi=150)
    print(f"\nFigure -> {out}/phase2_deltaofv_calibration.png")

    return verdict_ok


# %% ------------------------------------------------------------- config
# NOTE: unlike the other scripts in this project, this file DOES use
# `if __name__ == "__main__":` below, breaking the pure cell-by-cell pattern
# used elsewhere. This is not a style choice -- it's a hard requirement.
# ProcessPoolExecutor (used for --n-workers > 1) spawns new processes that
# re-import this file to find _fit_one_replicate. On macOS, spawn is the
# default start method, which means each worker executes this module's
# top-level code up to (but not including) a __main__ guard. Without one,
# every worker would re-parse argv and re-run the entire experiment,
# recursively spawning its own pool of workers -- an immediate crash or
# runaway recursion, not just an inefficiency. This guard is what makes
# --n-workers actually work rather than corrupting the run.
# Function/class definitions above are still safe to run cell-by-cell
# interactively; only this final block needs `python nlme_vi_phase2_deltaofv.py`
# (or `uv run ...`) rather than individual cell execution.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100,
                    help="more reps = more power to detect miscalibration; "
                         "the KS test needs at least ~30-50 positive-dOFV reps")
    ap.add_argument("--subjects", type=int, default=120)
    ap.add_argument("--K", type=int, default=64,
                    help="use the best-corrected VI arm (high K) for this check "
                         "-- calibration should be tested on the arm you'd "
                         "actually report, not the known-biased K=1 arm")
    ap.add_argument("--max-steps", type=int, default=25000)
    ap.add_argument("--n-workers", type=int, default=1,
                    help="replicates are fully independent -- run this many in "
                         "parallel worker processes. Try os.cpu_count()-1 on a "
                         "multi-core machine. Default 1 = original sequential "
                         "behavior.")
    ap.add_argument("--eval-k", type=int, default=None,
                    help="override is_marginal_loglik's post-hoc evaluation K "
                         "(default 4000, set in nlmevi_core). This is the actual "
                         "target of the IS-estimation-noise hypothesis -- raising "
                         "THIS, not --K (training), is the real test of whether "
                         "finite-sample IS noise explains the dOFV miscalibration.")
    ap.add_argument("--posterior", choices=["free", "amortized"], default="free",
                    help="free (default) gives the full model N extra free "
                         "(mu,log_s) parameter-pairs vs the reduced model -- far "
                         "more than the single population-level Omega_ka degree "
                         "of freedom the Self-Liang reference assumes. amortized "
                         "shares a single encoder across subjects instead, so the "
                         "full model only adds one output dimension, not N extra "
                         "parameter pairs. If amortized calibrates meaningfully "
                         "better than free, that confirms the effective-DoF "
                         "explanation for the observed miscalibration.")
    ap.add_argument("--out", default="outputs")
    args, _ = ap.parse_known_args()

    N_REPS, N_SUBJ, K = args.reps, args.subjects, args.K
    MAX_STEPS, OUT = args.max_steps, args.out
    N_WORKERS = args.n_workers
    EVAL_K = args.eval_k if args.eval_k is not None else 4000
    POSTERIOR = args.posterior
    print(f"config: N_SUBJ={N_SUBJ}  N_REPS={N_REPS}  K={K}  MAX_STEPS={MAX_STEPS}  "
         f"N_WORKERS={N_WORKERS}  EVAL_K={EVAL_K}  POSTERIOR={POSTERIOR}")

    # %% --------------------------------------------------------- experiment
    print("=" * 72)
    print(f"dOFV NULL CALIBRATION  N={N_SUBJ}  reps={N_REPS}  K={K}  eval_k={EVAL_K}  "
         f"posterior={POSTERIOR}")
    print("True model: om_Ka = 0 (reduced model is correctly specified)")
    print("=" * 72)

    df = run_null_calibration(N_REPS, N_SUBJ, K, MAX_STEPS, OUT,
                              n_workers=N_WORKERS, eval_k=EVAL_K, posterior=POSTERIOR)
    df.to_csv(f"{OUT}/phase2_deltaofv_results.csv", index=False)

    # %% ------------------------------------------------------ calibrate
    calibrate(df, OUT)
    print(f"\nRaw results -> {OUT}/phase2_deltaofv_results.csv")