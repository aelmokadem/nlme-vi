"""
Phase 2 / PSIS: does the per-subject trust diagnostic actually catch bad fits?
================================================================================

THE QUESTION THIS SCRIPT ANSWERS:

    nlmevi_core.is_marginal_loglik returns, per subject, an effective sample
    size (ess) and a max-weight-share proxy (top_share) for the importance
    sampling estimate. The claim is that these flag when a subject's fitted
    q(eta) is NOT a trustworthy proposal for that subject's true posterior --
    a diagnostic with no counterpart in classical FOCE/SAEM output.

    That claim has never been demonstrated. This script demonstrates it: a
    GOOD fit (converged properly) and a BAD fit (deliberately under-trained,
    same data) on the same subjects, showing the diagnostic separates them.

WHY UNDER-TRAINING IS THE RIGHT "BAD" CASE TO USE HERE
    PSIS/ESS diagnoses PROPOSAL QUALITY -- whether q(eta) matches the shape
    of the true posterior p(eta|y,theta) closely enough for importance
    sampling to work -- not point-estimate accuracy per se. Stopping training
    early is exactly the failure mode that compromises proposal quality
    without necessarily making the fitted (mu, log_s) look obviously wrong at
    a glance, which makes it a fair, non-strawman test of whether the
    diagnostic is actually doing its job (vs. only catching things that were
    already obvious from the parameter estimates).

REQUIRES  nlmevi_core.py in the same directory.
"""

# %% ---------------------------------------------------------------- imports
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

from nlmevi_core import TrueParams, OneCmtOral, simulate, fit_model

torch.set_default_dtype(torch.float64)


# %% ------------------------------------------------------------ experiment
def run_comparison(n_subj, seed, out, bad_steps=30):
    tp = TrueParams()
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
    data = simulate(tp, n_subj, times, seed=seed)   # SAME data for both arms
    model = OneCmtOral(tp.dose)
    theta_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                 math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]

    print("Fitting GOOD arm (adaptive, full convergence)...")
    theta_g, q_g, ll_g, ess_g, top_g, n_run_g, conv_g, _ = fit_model(
        model, data, "free", "gaussian", K=8, theta_init=theta_init, n_eta=3,
        adaptive=True, max_steps=25000,
    )
    print(f"  converged={conv_g} at step {n_run_g}, LL={ll_g:.1f}")

    print(f"Fitting BAD arm (fixed {bad_steps}-step under-training, same data)...")
    theta_b, q_b, ll_b, ess_b, top_b, n_run_b, conv_b, _ = fit_model(
        model, data, "free", "gaussian", K=8, theta_init=theta_init, n_eta=3,
        adaptive=False, n_steps=bad_steps,
    )
    print(f"  ran {n_run_b} steps (deliberately short), LL={ll_b:.1f}")

    df = pd.DataFrame({
        "subject": np.arange(n_subj).tolist() * 2,
        "arm": ["good"] * n_subj + ["bad"] * n_subj,
        "ess": np.concatenate([ess_g, ess_b]),
        "top_share": np.concatenate([top_g, top_b]),
    })
    df.to_csv(f"{out}/phase2_psis_results.csv", index=False)
    return df, (ll_g, ll_b)


def analyze(df, lls, out, K_eval=4000):
    ll_g, ll_b = lls
    print("\n" + "=" * 72)
    print("DOES THE DIAGNOSTIC SEPARATE THE TWO ARMS?")
    print("=" * 72)
    print(f"  Overall marginal LL:  good={ll_g:.1f}   bad={ll_b:.1f}   "
          f"(bad should be visibly worse -- this alone is the classical check)")

    print(f"\n  Per-subject ESS (out of {K_eval} importance samples):")
    summ = df.groupby("arm")[["ess", "top_share"]].agg(["mean", "median", "min"])
    print(summ.round(1).to_string())

    frac_low_ess = df.groupby("arm").apply(
        lambda g: (g.ess < 0.05 * K_eval).mean(), include_groups=False)
    frac_high_top = df.groupby("arm").apply(
        lambda g: (g.top_share > 0.3).mean(), include_groups=False)
    print(f"\n  Fraction of subjects with ESS < 5% of K (proposal essentially failing):")
    print(f"    {frac_low_ess.to_string()}")
    print(f"  Fraction of subjects with top_share > 0.3 (one sample dominates):")
    print(f"    {frac_high_top.to_string()}")

    good_ess = df[df.arm == "good"].ess.mean()
    bad_ess = df[df.arm == "bad"].ess.mean()
    separates = bad_ess < 0.7 * good_ess   # bad arm should show meaningfully lower ESS

    print("\n" + "-" * 72)
    if separates:
        print("VERDICT: the diagnostic DOES separate the arms -- bad ESS shows up")
        print("even though this is a purely proposal-quality signal, independent of")
        print("whether the point estimates alone would have tipped you off.")
    else:
        print("VERDICT: ESS did NOT meaningfully differ between arms. Either the")
        print("under-training wasn't severe enough to compromise the proposal (try a")
        print("smaller --bad-steps), or the diagnostic isn't sensitive enough as")
        print("implemented -- worth comparing against a real arviz.psislw k-hat here.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arm, color in zip(["good", "bad"], ["tab:blue", "tab:red"]):
        sub = df[df.arm == arm]
        axes[0].hist(sub.ess, bins=20, alpha=0.6, label=arm, color=color)
        axes[1].hist(sub.top_share, bins=20, alpha=0.6, label=arm, color=color)
    axes[0].set_xlabel("effective sample size")
    axes[0].set_title("Per-subject ESS")
    axes[1].set_xlabel("max importance-weight share")
    axes[1].set_title("Per-subject top_share (k-hat proxy)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Does the trust diagnostic separate a converged fit from an under-trained one?")
    fig.tight_layout()
    fig.savefig(f"{out}/phase2_psis_comparison.png", dpi=150)
    print(f"\nFigure -> {out}/phase2_psis_comparison.png")
    return separates


# %% ------------------------------------------------------------- config
ap = argparse.ArgumentParser()
ap.add_argument("--subjects", type=int, default=120)
ap.add_argument("--bad-steps", type=int, default=30,
                help="how few steps the deliberately-bad arm gets")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="outputs")
args, _ = ap.parse_known_args()

N_SUBJ, BAD_STEPS, SEED, OUT = args.subjects, args.bad_steps, args.seed, args.out
print(f"config: N_SUBJ={N_SUBJ}  BAD_STEPS={BAD_STEPS}  SEED={SEED}")


# %% --------------------------------------------------------- experiment
print("=" * 72)
print(f"PSIS/ESS DIAGNOSTIC VALIDATION  N={N_SUBJ}  bad_steps={BAD_STEPS}")
print("=" * 72)

df, lls = run_comparison(N_SUBJ, SEED, OUT, BAD_STEPS)
# `df` (per-subject ess/top_share for both arms) and `lls` (overall marginal
# LL for each arm) are now in your interactive session.


# %% ------------------------------------------------------------ analyze
analyze(df, lls, OUT)
print(f"\nRaw results -> {OUT}/phase2_psis_results.csv")
