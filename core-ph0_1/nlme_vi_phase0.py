"""
Phase 0 go/no-go: does variational inference systematically shrink Omega in NLME?
================================================================================

THE QUESTION THIS SCRIPT ANSWERS (and nothing else):

    Multiple independent groups have reported, in passing, that variational
    estimators for NLME models return random-effect variances that are too
    small. Nobody had established this on known ground truth or shown what
    drives it before this project.

    This script asks: on a simple 1-compartment PK model where we know the
    true Omega exactly, (a) does the shrinkage appear, and (b) does
    importance weighting remove it?

DESIGN
    Generative model     1-cmt oral PK, analytic solution (OneCmtOral, from
                         nlmevi_core -- no ODE solver needed)
    Random effects       eta on log CL, log V, log ka; diagonal Omega
    Inference arms       {free per-subject q, amortized encoder q} x {K = 1, 8, 64}
    Replicates           independent datasets; EVERY ARM SEES BYTE-IDENTICAL DATA

STRUCTURE NOTE
    All shared machinery (models, posteriors, iw_elbo, is_marginal_loglik,
    the adaptive trainer) lives in nlmevi_core.py, imported below. This
    script only contains what's specific to the Phase 0 question: sanity
    checks, the experiment loop, and the go/no-go verdict. Phase 1, 2, and 3
    scripts import the same core rather than this file.

THE ONE METHODOLOGICAL RULE
    One seed generates one replicate's data. Nothing about the inference
    method touches the data-generating RNG.

RUN
    python nlme_vi_phase0.py --quick      # ~1 min,  smoke test
    python nlme_vi_phase0.py              # full run
"""

# %% ---------------------------------------------------------------- imports
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlmevi_core import (
    TrueParams, conc_analytic, OneCmtOral, simulate, to_tensors,
    FreePosterior, iw_elbo, is_marginal_loglik, fit_model,
    LINEAR_PARAM_NAMES as PARAM_NAMES, true_vector_linear as true_vector,
)

torch.set_default_dtype(torch.float64)


# %% ------------------------------------------------------------ diagnostics
def sanity_checks(tp):
    """
    Tests that must pass before ANY result below is believable.
    The simulate-and-recover / closed-form-likelihood test is the
    highest-value check in the project.
    """
    print("=" * 72)
    print("SANITY CHECKS")
    print("=" * 72)
    ok = True

    # 1. Analytic solution == numerical ODE solution
    from scipy.integrate import solve_ivp
    CL, V, ka, dose = 3.0, 30.0, 1.2, 100.0
    tt = np.array([0.5, 1, 2, 4, 8, 24.0])

    def rhs(t, y):
        return [-ka * y[0], ka * y[0] - (CL / V) * y[1]]

    sol = solve_ivp(rhs, [0, 24], [dose, 0.0], t_eval=tt, rtol=1e-10, atol=1e-12)
    num = sol.y[1] / V
    ana = conc_analytic(torch.tensor(tt), torch.tensor(CL), torch.tensor(V),
                        torch.tensor(ka), dose).numpy()
    err = np.max(np.abs(num - ana))
    print(f"  [1] analytic vs ODE solver      max abs err = {err:.2e}   "
          f"{'PASS' if err < 1e-6 else 'FAIL'}")
    ok &= err < 1e-6

    # 2/3. K=1 IW-ELBO equals plain ELBO; IW-ELBO increases in K
    model = OneCmtOral(tp.dose)
    data = simulate(tp, 40, [0.5, 1, 2, 4, 8, 24], seed=999)
    batch = to_tensors(data)
    q = FreePosterior(40, 3)
    theta = torch.tensor([math.log(3.0), math.log(30.0), math.log(1.2),
                          math.log(.3), math.log(.2), math.log(.4), math.log(.2)])

    torch.manual_seed(0); a = iw_elbo(model, q, theta, batch, 1).item()
    torch.manual_seed(0); b = iw_elbo(model, q, theta, batch, 1).item()
    print(f"  [2] K=1 IW-ELBO is the ELBO     {a:.6f} == {b:.6f}   "
          f"{'PASS' if abs(a - b) < 1e-9 else 'FAIL'}")
    ok &= abs(a - b) < 1e-9

    vals = []
    for K in (1, 4, 16, 64, 256):
        torch.manual_seed(7)
        vals.append(np.mean([iw_elbo(model, q, theta, batch, K).item() for _ in range(8)]))
    mono = all(vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1))
    print(f"  [3] IW-ELBO monotone in K       {[f'{v:.1f}' for v in vals]}   "
          f"{'PASS' if mono else 'FAIL'}")
    ok &= mono

    # 4. IS marginal likelihood matches a CLOSED-FORM linear Gaussian case.
    #    If this fails, every likelihood number downstream is noise.
    torch.manual_seed(3)
    n, T, s_eta, s_eps = 30, 5, 0.7, 0.4
    X = torch.randn(n, T)
    y = X * 0 + torch.randn(n, 1) * s_eta + torch.randn(n, T) * s_eps
    Sig = s_eta ** 2 * torch.ones(T, T) + s_eps ** 2 * torch.eye(T)
    exact = torch.distributions.MultivariateNormal(torch.zeros(T), Sig).log_prob(y).sum().item()

    class LMM:
        n_eta = 1
        def log_prior(self, eta, th):
            return (-0.5 * (eta / s_eta) ** 2 - math.log(s_eta) - 0.5 * math.log(2 * math.pi)).sum(-1)
        def log_lik(self, yy, eta, th, t, m):
            r = yy - eta
            return (-0.5 * (r / s_eps) ** 2 - math.log(s_eps) - 0.5 * math.log(2 * math.pi)).sum(-1)
        def log_joint(self, yy, eta, th, t, m):
            return self.log_prior(eta, th) + self.log_lik(yy, eta, th, t, m)

    prec = 1 / s_eta ** 2 + T / s_eps ** 2
    post_mu = (y.sum(1, keepdim=True) / s_eps ** 2) / prec
    post_sd = (1 / prec) ** 0.5
    qe = FreePosterior(n, 1)
    with torch.no_grad():
        qe.mu.copy_(post_mu); qe.log_s.fill_(math.log(post_sd))
    est, _, _ = is_marginal_loglik(LMM(), qe, None,
                                   {"logy": y, "t": torch.zeros(n, T), "mask": torch.ones(n, T)},
                                   K=20000)
    d = abs(est - exact)
    print(f"  [4] IS marginal LL vs closed form  {est:.4f} vs {exact:.4f}  "
          f"|diff|={d:.4f}   {'PASS' if d < 0.05 else 'FAIL'}")
    ok &= d < 0.05

    print(f"\n  {'ALL CHECKS PASSED' if ok else '*** CHECKS FAILED -- STOP ***'}\n")
    return ok


# %% ------------------------------------------------------------ experiment
def run_experiment(tp, n_subj, times, n_reps, K_grid, kinds, n_steps, drep=True,
                   max_steps=25000):
    rows = []
    truth = true_vector(tp)

    for rep in range(n_reps):
        # ONE seed -> ONE dataset. Every arm below reads this identical object.
        data = simulate(tp, n_subj, times, seed=1000 + rep)
        model = OneCmtOral(tp.dose)
        theta_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                     math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]

        for kind in kinds:
            for K in K_grid:
                t0 = time.time()
                theta, q, ll, ess, top, n_run, converged = fit_model(
                    model, data, kind, "gaussian", K, theta_init, n_eta=3,
                    n_steps=n_steps, drep=drep, seed=rep, max_steps=max_steps,
                )
                est = np.concatenate([np.exp(theta[:3]), np.exp(theta[3:6]), [np.exp(theta[6])]])
                for j, name in enumerate(PARAM_NAMES):
                    rows.append(dict(
                        replicate=rep, seed=1000 + rep, posterior=kind, K=K,
                        param=name, estimate=est[j], truth=truth[j],
                        rel_bias_pct=100 * (est[j] - truth[j]) / truth[j],
                        marginal_ll=ll, mean_ess=float(ess.mean()),
                        frac_bad=float((top > 0.5).mean()),
                        n_steps_run=n_run, converged=converged,
                        secs=time.time() - t0,
                    ))
                flag = "" if converged else "  *** DID NOT CONVERGE ***"
                print(f"  rep {rep:2d} | {kind:9s} K={K:3d} | "
                      f"om_CL {est[3]:.3f} om_V {est[4]:.3f} om_ka {est[5]:.3f} | "
                      f"LL {ll:8.1f} | steps={n_run:6d} | {time.time()-t0:5.1f}s{flag}")
    return pd.DataFrame(rows)


def summarize(df):
    om = df[df.param.str.startswith("om_")]
    tab = (om.groupby(["posterior", "K", "param"])
             .rel_bias_pct.agg(["mean", "std", "count"])
             .reset_index()
             .rename(columns={"mean": "rel_bias_%", "std": "sd", "count": "n"}))

    print("\n" + "=" * 72)
    print("HEADLINE: relative bias (%) in the random-effect SDs")
    print("  negative = SHRINKAGE = the phenomenon we are testing for")
    print("=" * 72)
    piv = tab.pivot_table(index=["posterior", "K"], columns="param", values="rel_bias_%")
    print(piv.round(2).to_string())

    print("\n  Averaged over the three omegas:")
    avg = om.groupby(["posterior", "K"]).rel_bias_pct.mean().round(2)
    print(avg.to_string())

    print("\n" + "-" * 72)
    print("Fixed effects (for contrast -- expected to be nearly unbiased):")
    fe = df[df.param.isin(["CL", "V", "ka"])]
    print(fe.groupby(["posterior", "K"]).rel_bias_pct.mean().round(2).to_string())

    print("\n" + "-" * 72)
    print("Marginal log-likelihood (IS estimate, higher = better fit):")
    print(df.groupby(["posterior", "K"]).marginal_ll.mean().round(1).to_string())
    return tab


def make_figure(df, path):
    om = df[df.param.str.startswith("om_")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, kind in zip(axes, sorted(om.posterior.unique())):
        sub = om[om.posterior == kind]
        for p in ["om_CL", "om_V", "om_ka"]:
            s = sub[sub.param == p].groupby("K").rel_bias_pct.agg(["mean", "sem"])
            ax.errorbar(s.index, s["mean"], yerr=s["sem"], marker="o",
                        capsize=3, label=p)
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (importance samples);  K=1 is the plain ELBO")
        ax.set_title(f"{kind} posterior")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative bias in omega (%)")
    axes[0].legend()
    fig.suptitle("Does importance weighting remove the variance-component shrinkage?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nFigure -> {path}")


# %% ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--subjects", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=25000,
                    help="safety cap for adaptive convergence-based training")
    ap.add_argument("--no-drep", action="store_true",
                    help="disable doubly-reparameterized gradients")
    ap.add_argument("--out", default="/mnt/user-data/outputs")
    args = ap.parse_args()

    tp = TrueParams()
    TIMES = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]   # sparse-ish, 6 samples/subject

    if args.quick:
        n_reps, n_steps, n_subj, K_grid = 2, 800, 60, [1, 16]
    else:
        n_reps, n_steps, n_subj, K_grid = 8, 4000, 120, [1, 8, 64]
    n_reps = args.reps or n_reps
    n_steps = args.steps or n_steps
    n_subj = args.subjects or n_subj

    if not sanity_checks(tp):
        raise SystemExit("Sanity checks failed -- do not trust anything below.")

    print("=" * 72)
    print(f"EXPERIMENT  N={n_subj} subjects x {len(TIMES)} obs, "
          f"{n_reps} replicates, K in {K_grid}, drep={not args.no_drep}")
    print(f"TRUE omegas: CL={tp.om_CL}  V={tp.om_V}  ka={tp.om_ka}")
    print("=" * 72)

    df = run_experiment(tp, n_subj, TIMES, n_reps, K_grid,
                        ["free", "amortized"], n_steps, drep=not args.no_drep,
                        max_steps=args.max_steps)

    tab = summarize(df)
    df.to_csv(f"{args.out}/phase0_results.csv", index=False)
    make_figure(df, f"{args.out}/phase0_omega_bias.png")
    print(f"Raw results -> {args.out}/phase0_results.csv")

    n_bad = (~df.drop_duplicates(["replicate", "posterior", "K"]).converged).sum()
    n_total = df.drop_duplicates(["replicate", "posterior", "K"]).shape[0]
    print(f"\nConvergence: {n_total - n_bad}/{n_total} fits converged within "
          f"max_steps. {'*** '+str(n_bad)+' DID NOT -- see n_steps_run/converged '
          'columns in the CSV, consider raising --max-steps ***' if n_bad else ''}")

    # ---- the go/no-go verdict -------------------------------------------
    om = df[df.param.str.startswith("om_")]
    base = om[(om.K == min(K_grid))].rel_bias_pct.mean()
    top = om[(om.K == max(K_grid))].rel_bias_pct.mean()
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  mean omega bias at K={min(K_grid):<3d} (plain ELBO) : {base:+.2f}%")
    print(f"  mean omega bias at K={max(K_grid):<3d} (IW-ELBO)    : {top:+.2f}%")
    if base < -1.0:
        print("\n  -> Shrinkage REPRODUCES on known ground truth. Aim 1 is real.")
        if top > base + 0.5:
            print("  -> Importance weighting reduces it. Aim 2 mechanism confirmed.")
        else:
            print("  -> IW does NOT fix it. The variational FAMILY is the likely")
            print("     culprit, not the bound.")
    else:
        print("\n  -> No meaningful shrinkage here. Before abandoning: try sparser")
        print("     sampling, larger true omegas, and a nonlinear model.")
