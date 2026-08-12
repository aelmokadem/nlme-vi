"""
Phase 1: stress-testing the Omega-shrinkage finding from Phase 0
==================================================================

Phase 1 asks four follow-up questions, each addressing a specific way the
Phase 0 result could be an artifact rather than a real phenomenon:

  Q1. CONVERGENCE   Is the K=1 bias a converged answer, or just an
                     under-trained one? --converge-check trains a single
                     replicate until the omega estimates plateau (or a
                     safety cap is hit) and plots the trajectory.

  Q2. IDENTIFIABILITY  Shrinkage should be WORSE when the individual
                        posterior is less informed by data. Sparse
                        sampling (3 obs/subject instead of 6) tests this.

  Q3. NONLINEARITY   Does the effect survive when the structural model is
                      genuinely nonlinear? Michaelis-Menten elimination
                      (OneCmtIVBolusMM, from nlmevi_core) is the tier for
                      this -- no closed-form solution, RK4-integrated.

  Q4. VARIATIONAL FAMILY   Does a richer family (conditional normalizing
                            flow instead of diagonal Gaussian) close bias
                            that increasing K alone leaves on the table?

STRUCTURE NOTE
    All shared machinery -- models, posteriors (including the flow), iw_elbo,
    is_marginal_loglik, and the adaptive trainer with best-checkpoint
    tracking -- lives in nlmevi_core.py. This script only contains what's
    specific to Phase 1: the four scenarios, the convergence-check
    diagnostic, and the summary/figure logic for this particular grid.

THE ONE RULE CARRIED OVER FROM PHASE 0, UNCHANGED:
    One seed -> one dataset. Every method arm in a given (scenario, replicate)
    cell reads byte-identical data.

USAGE
    # Q1 first -- cheap, and the one thing that could invalidate everything else
    python nlme_vi_phase1.py --converge-check

    # Q2 + Q4 (default, dense+sparse linear scenarios, gaussian+flow families)
    python nlme_vi_phase1.py

    # Q3 -- add the expensive nonlinear tier explicitly, with cost caps
    python nlme_vi_phase1.py --scenarios dense,sparse,nonlinear \\
        --nl-reps 4 --nl-steps 1500 --nl-subjects 60

REQUIRES  nlmevi_core.py in the same directory.
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
    TrueParams, OneCmtOral, simulate, to_tensors,
    MMTrueParams, build_grid, OneCmtIVBolusMM, simulate_mm, mm_true_vector,
    MM_PARAM_NAMES, LINEAR_PARAM_NAMES, true_vector_linear,
    iw_elbo, train_to_convergence, fit_model, make_posterior,
)

torch.set_default_dtype(torch.float64)


# %% =================================================================
#  Q1: convergence check -- is K=1's bias real, or just under-trained?
# =====================================================================
def converge_check(max_steps=25000, log_every=250, n_subj=120, seed=0, out=".",
                   drep=True, min_steps=2000, tol=0.01, patience=4):
    """
    Trains K=1 (plain ELBO, free posterior) on ONE replicate until the omega
    estimates THEMSELVES plateau (via train_to_convergence), not for a fixed
    step count. Records the full trajectory at every `log_every` steps for
    the plot regardless of when it stops.

    Read the resulting figure like this:
      - flat plateau clearly below/above the true dashed line -> real bias,
        and the verdict below will say CONVERGED.
      - still visibly moving when max_steps is hit -> genuinely needs a
        higher --converge-steps; the verdict will say NOT CONVERGED rather
        than silently reporting a mid-drift snapshot as if it were final.
    """
    tp = TrueParams()
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
    data = simulate(tp, n_subj, times, seed=2000 + seed)
    model = OneCmtOral(tp.dose)
    batch = to_tensors(data)

    q = make_posterior("free", "gaussian", n_subj, len(times), 3)
    theta_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                 math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]
    theta = torch.tensor(theta_init, requires_grad=True)
    opt = torch.optim.Adam([{"params": [theta], "lr": 0.05},
                            {"params": q.parameters(), "lr": 0.05}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.7)

    torch.manual_seed(seed)
    rows = []
    truth = true_vector_linear(tp)
    t0 = time.time()
    _n = {"s": 0}

    def step_fn():
        opt.zero_grad()
        loss = -iw_elbo(model, q, theta, batch, K=1, drep=drep) / n_subj
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta] + list(q.parameters()), 10.0)
        opt.step()
        sched.step()
        if _n["s"] % log_every == 0:
            with torch.no_grad():
                om = torch.exp(theta[3:6]).numpy()
            rows.append(dict(step=_n["s"], om_CL=om[0], om_V=om[1], om_ka=om[2],
                             loss=loss.item(), secs=time.time() - t0))
            print(f"  step {_n['s']:6d} | om_CL {om[0]:.3f} om_V {om[1]:.3f} "
                  f"om_ka {om[2]:.3f} | loss {loss.item():.4f}")
        _n["s"] += 1

    def get_omega_fn():
        with torch.no_grad():
            return torch.exp(theta[3:6]).cpu().numpy()

    n_run, converged, _ = train_to_convergence(
        step_fn, get_omega_fn, min_steps=min_steps, check_every=log_every,
        patience=patience, tol=tol, max_steps=max_steps, verbose=True,
    )

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/phase1_convergence.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, truth_val in zip(["om_CL", "om_V", "om_ka"], truth[3:6]):
        ax.plot(df.step, df[name], label=name)
        ax.axhline(truth_val, ls="--", alpha=0.5, color=ax.lines[-1].get_color())
    ax.set_xlabel("optimization step")
    ax.set_ylabel("omega estimate")
    status = "CONVERGED" if converged else "DID NOT CONVERGE -- raise --converge-steps"
    ax.set_title(f"K=1 convergence check (stopped at step {n_run}, {status})\n"
                 f"dashed = truth; plateau (either side of truth) = the real answer")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/phase1_convergence.png", dpi=150)
    print(f"\nSaved: {out}/phase1_convergence.csv, {out}/phase1_convergence.png")

    print(f"\nVerdict: {'CONVERGED' if converged else 'DID NOT CONVERGE'} at step {n_run}")
    if not converged:
        print(f"  Ran the full max_steps={max_steps} without the plateau test "
              f"passing. Re-run with a higher --converge-steps.")
    final_om = get_omega_fn()
    for name, val, t in zip(["om_CL", "om_V", "om_ka"], final_om, truth[3:6]):
        print(f"  {name}: final estimate {val:.3f}  (truth {t:.3f}, "
              f"rel. bias {100*(val-t)/t:+.1f}%)")


# %% =================================================================
#  Q2 + Q3 + Q4: the combined scenario x family x posterior x K grid
# =====================================================================
def get_scenario(name, args):
    """
    Returns (model, data_fn, theta_init, param_names, truth_vector, n_subj, n_steps)
    for one named scenario. data_fn(seed) -> data dict.
    """
    if name == "dense":
        tp = TrueParams()
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
        model = OneCmtOral(tp.dose)
        theta_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                     math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]
        return (model, lambda seed: simulate(tp, args.subjects, times, seed),
               theta_init, LINEAR_PARAM_NAMES, true_vector_linear(tp),
               args.subjects, args.steps)

    if name == "sparse":
        tp = TrueParams()
        times = [1.0, 4.0, 24.0]     # 3 obs/subject instead of 6 -- Q2
        model = OneCmtOral(tp.dose)
        theta_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                     math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]
        return (model, lambda seed: simulate(tp, args.subjects, times, seed),
               theta_init, LINEAR_PARAM_NAMES, true_vector_linear(tp),
               args.subjects, args.steps)

    if name == "nonlinear":
        mp = MMTrueParams()
        times = [0.5, 2.0, 8.0, 24.0]      # kept short: this tier is expensive
        grid, obs_idx = build_grid(times, dt_max=args.mm_dt)
        model = OneCmtIVBolusMM(mp.dose, grid, obs_idx)
        theta_init = [math.log(6.0), math.log(3.0), math.log(25.0),
                     math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]
        return (model, lambda seed: simulate_mm(mp, args.nl_subjects, times, args.mm_dt, seed),
               theta_init, MM_PARAM_NAMES, mm_true_vector(mp),
               args.nl_subjects, args.nl_steps)

    raise ValueError(name)


def run_grid(args):
    scenarios = args.scenarios.split(",")
    families = args.families.split(",")
    posteriors = args.posteriors.split(",")
    K_grid = [int(k) for k in args.K.split(",")]

    rows = []
    for scen_name in scenarios:
        model, data_fn, theta_init, names, truth, n_subj, n_steps = get_scenario(scen_name, args)
        n_reps = args.nl_reps if scen_name == "nonlinear" else args.reps
        if scen_name == "nonlinear":
            print(f"\n[nonlinear tier is expensive: N={n_subj}, reps={n_reps}, "
                  f"grid dt={args.mm_dt} -- override with --nl-* flags]")

        for rep in range(n_reps):
            seed = 3000 + rep
            data = data_fn(seed)   # ONE dataset per (scenario, rep); every arm below shares it

            for kind in posteriors:
                for family in families:
                    for K in K_grid:
                        t0 = time.time()
                        # mc_reps: only the flow family at small K showed
                        # gradient-noise instability (amortized+flow+K=1).
                        # Gaussian never needed it; K>=8 already gets
                        # averaging for free from the logsumexp itself.
                        mc_reps = args.mc_reps_k1 if (family == "flow" and K == 1) else 1
                        theta, q, ll, ess, top, n_run, converged = fit_model(
                            model, data, kind, family, K, theta_init,
                            n_eta=3, n_steps=n_steps, drep=not args.no_drep,
                            seed=rep, mc_reps=mc_reps, max_steps=args.max_steps,
                        )
                        est = np.concatenate([np.exp(theta[:3]), np.exp(theta[3:6]), [np.exp(theta[6])]])
                        for j, pname in enumerate(names):
                            rows.append(dict(
                                scenario=scen_name, replicate=rep, seed=seed,
                                posterior=kind, family=family, K=K, param=pname,
                                estimate=est[j], truth=truth[j],
                                rel_bias_pct=100 * (est[j] - truth[j]) / truth[j],
                                marginal_ll=ll, mean_ess=float(ess.mean()),
                                n_steps_run=n_run, converged=converged,
                                secs=time.time() - t0,
                            ))
                        flag = "" if converged else "  *** DID NOT CONVERGE ***"
                        print(f"  [{scen_name:9s}] rep {rep:2d} | {kind:9s}/{family:8s} K={K:3d} | "
                              f"om avg est {est[3:6].mean():.3f} (truth {truth[3:6].mean():.3f}) | "
                              f"LL {ll:8.1f} | steps={n_run:6d} | {time.time()-t0:5.1f}s{flag}")

    return pd.DataFrame(rows)


def summarize_v2(df):
    dedup = df.drop_duplicates(["scenario", "replicate", "posterior", "family", "K"])
    n_bad = (~dedup.converged).sum()
    n_total = len(dedup)
    print("=" * 78)
    print(f"Convergence: {n_total - n_bad}/{n_total} fits converged within max_steps.")
    if n_bad:
        print(f"*** {n_bad} DID NOT converge -- see converged/n_steps_run columns in "
              f"the CSV. Bias numbers for those cells are not trustworthy as-is; "
              f"consider raising --max-steps. ***")
        print(dedup[~dedup.converged][["scenario", "posterior", "family", "K", "n_steps_run"]]
             .to_string(index=False))

    om = df[df.param.str.startswith("om_")]
    print("\n" + "=" * 78)
    print("Mean relative bias (%) in omega, by scenario / posterior / family / K")
    print("=" * 78)
    piv = (om.groupby(["scenario", "posterior", "family", "K"])
             .rel_bias_pct.mean().unstack("K").round(2))
    print(piv.to_string())

    print("\n" + "-" * 78)
    print("Q2 check -- does sparse sampling worsen shrinkage relative to dense, at matched K?")
    if {"dense", "sparse"}.issubset(set(df.scenario.unique())):
        cmp = (om[om.scenario.isin(["dense", "sparse"])]
              .groupby(["scenario", "K"]).rel_bias_pct.mean().unstack("scenario").round(2))
        print(cmp.to_string())
    else:
        print("  (run both 'dense' and 'sparse' scenarios to see this comparison)")

    print("\n" + "-" * 78)
    print("Q4 check -- does the flow family close the gap gaussian leaves at high K?")
    if {"gaussian", "flow"}.issubset(set(df.family.unique())):
        cmp = (om.groupby(["family", "K"]).rel_bias_pct.mean().unstack("family").round(2))
        print(cmp.to_string())
    else:
        print("  (run both 'gaussian' and 'flow' families to see this comparison)")

    if "nonlinear" in df.scenario.unique():
        print("\n" + "-" * 78)
        print("Q3 check -- does shrinkage survive in the nonlinear (MM) scenario?")
        nl = om[om.scenario == "nonlinear"]
        print(nl.groupby(["posterior", "family", "K"]).rel_bias_pct.mean().round(2).to_string())

    return piv


def make_figure_v2(df, path):
    om = df[df.param.str.startswith("om_")]
    scenarios = sorted(om.scenario.unique())
    families = sorted(om.family.unique())
    fig, axes = plt.subplots(len(scenarios), len(families),
                             figsize=(6 * len(families), 4 * len(scenarios)),
                             squeeze=False, sharey="row")

    for i, scen in enumerate(scenarios):
        for j, fam in enumerate(families):
            ax = axes[i][j]
            sub = om[(om.scenario == scen) & (om.family == fam)]
            for kind in sorted(sub.posterior.unique()):
                s2 = sub[sub.posterior == kind]
                agg = s2.groupby("K").rel_bias_pct.agg(["mean", "sem"])
                ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"], marker="o",
                           capsize=3, label=kind)
            ax.axhline(0, color="k", lw=1, ls="--")
            ax.set_xscale("log", base=2)
            ax.set_title(f"{scen} / {fam}")
            ax.grid(alpha=0.3)
            if i == len(scenarios) - 1:
                ax.set_xlabel("K")
            if j == 0:
                ax.set_ylabel("mean omega rel. bias (%)")
            if i == 0 and j == 0:
                ax.legend()

    fig.suptitle("Phase 1: omega bias across scenario x family x posterior x K", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nFigure -> {path}")


# %% ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--converge-check", action="store_true",
                    help="run Q1 only: convergence trace for a single replicate, then exit")
    ap.add_argument("--converge-steps", type=int, default=25000,
                    help="safety cap for the convergence check (stops earlier "
                         "on its own once omega plateaus)")

    ap.add_argument("--scenarios", default="dense,sparse",
                    help="comma list from {dense, sparse, nonlinear}")
    ap.add_argument("--families", default="gaussian,flow",
                    help="comma list from {gaussian, flow}")
    ap.add_argument("--posteriors", default="free,amortized",
                    help="comma list from {free, amortized}")
    ap.add_argument("--K", default="1,8,64")
    ap.add_argument("--max-steps", type=int, default=25000,
                    help="safety cap for adaptive convergence-based training; "
                         "NOT a fixed step count")
    ap.add_argument("--mc-reps-k1", type=int, default=8,
                    help="gradient-noise-reduction reps for flow family at K=1")

    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000,
                    help="fallback step count if --no-adaptive; ignored otherwise")
    ap.add_argument("--subjects", type=int, default=120)

    ap.add_argument("--nl-reps", type=int, default=4,
                    help="nonlinear tier is expensive -- kept small by default")
    ap.add_argument("--nl-steps", type=int, default=1500)
    ap.add_argument("--nl-subjects", type=int, default=60)
    ap.add_argument("--mm-dt", type=float, default=0.1, help="RK4 grid spacing (h)")

    ap.add_argument("--no-drep", action="store_true")
    ap.add_argument("--out", default="/mnt/user-data/outputs")
    args, _ = ap.parse_known_args()

    if args.converge_check:
        print("=" * 78)
        print("Q1: CONVERGENCE CHECK -- does K=1's bias plateau, or is it still moving?")
        print("=" * 78)
        converge_check(max_steps=args.converge_steps, out=args.out, drep=not args.no_drep)
        raise SystemExit(0)

    print("=" * 78)
    print(f"PHASE 1 GRID  scenarios={args.scenarios}  families={args.families}  "
          f"posteriors={args.posteriors}  K={args.K}")
    print("=" * 78)

    df = run_grid(args)
    df.to_csv(f"{args.out}/phase1_results.csv", index=False)
    summarize_v2(df)
    make_figure_v2(df, f"{args.out}/phase1_grid.png")
    print(f"\nRaw results -> {args.out}/phase1_results.csv")
