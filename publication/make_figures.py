"""
publication/make_figures.py -- manuscript figures from the project's result
CSVs. Companion to make_tables.py; same explicit-file-path design (see that
script's docstring for why -- several source scripts overwrite a fixed
default filename across different conditions, so this script never guesses
which file corresponds to which condition).

Figures deliberately have NO captions or descriptive titles baked into the
image (no fig.suptitle() with a "what this shows" sentence) -- only the
axis labels, legends, and panel labels needed for the plot to be
interpretable on its own. Caption text is left for the manuscript itself.

USAGE
    python publication/make_figures.py \
        --phase0-csv outputs/phase0_results.csv \
        --phase1-csv outputs/phase1_results.csv \
        --theoph-csv outputs/phase2_realdata_theoph.csv \
        --warfarin-csv outputs/phase2_realdata_warfarin.csv \
        --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
        --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized.csv \
        --psis-csv outputs/phase2_psis_results.csv \
        --out publication/figures

Any flag can be omitted; that figure is skipped, not an error.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def try_load(path, label):
    if not path:
        print(f"[skip] {label}: no path given")
        return None
    if not os.path.exists(path):
        print(f"[skip] {label}: {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"[loaded] {label}: {path} ({len(df)} rows)")
    return df


def savefig(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"-> {path}")


# %% ------------------------------------------------------------ figures
def fig_phase0(df, out_dir):
    """Omega bias vs K, one panel per posterior."""
    om = df[df.param.str.startswith("om_")]
    posteriors = sorted(om.posterior.unique())
    fig, axes = plt.subplots(1, len(posteriors), figsize=(5.5 * len(posteriors), 4.2),
                             sharey=True)
    if len(posteriors) == 1:
        axes = [axes]
    for ax, post in zip(axes, posteriors):
        sub = om[om.posterior == post]
        for p in sorted(sub.param.unique()):
            s = sub[sub.param == p].groupby("K").rel_bias_pct.agg(["mean", "sem"])
            ax.errorbar(s.index, s["mean"], yerr=s["sem"], marker="o", capsize=3, label=p)
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K")
        ax.set_title(post)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative bias in omega (%)")
    axes[0].legend()
    savefig(fig, "fig_phase0_omega_bias", out_dir)


def fig_phase1_grid(df, out_dir):
    """Omega bias vs K, grid of panels: scenario (rows) x family (cols),
    one line per posterior within each panel."""
    om = df[(df.param.str.startswith("om_")) & (df.scenario.isin(["dense", "sparse"]))]
    scenarios = sorted(om.scenario.unique())
    families = sorted(om.family.unique())
    fig, axes = plt.subplots(len(scenarios), len(families),
                             figsize=(5.5 * len(families), 4 * len(scenarios)),
                             squeeze=False, sharey="row")
    for i, scen in enumerate(scenarios):
        for j, fam in enumerate(families):
            ax = axes[i][j]
            sub = om[(om.scenario == scen) & (om.family == fam)]
            for post in sorted(sub.posterior.unique()):
                s = sub[sub.posterior == post].groupby("K").rel_bias_pct.agg(["mean", "sem"])
                ax.errorbar(s.index, s["mean"], yerr=s["sem"], marker="o",
                           capsize=3, label=post)
            ax.axhline(0, color="k", lw=1, ls="--")
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.3)
            if i == len(scenarios) - 1:
                ax.set_xlabel("K")
            if j == 0:
                ax.set_ylabel(f"{scen}\nomega bias (%)")
            ax.set_title(fam if i == 0 else "")
            if i == 0 and j == 0:
                ax.legend()
    savefig(fig, "fig_phase1_grid", out_dir)


def fig_nonlinear(df, out_dir):
    """Omega bias vs K for the nonlinear (MM) tier, one line per param."""
    nl = df[df.scenario == "nonlinear"] if "scenario" in df.columns else df
    om = nl[nl.param.str.startswith("om_")]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p in sorted(om.param.unique()):
        s = om[om.param == p].groupby("K").rel_bias_pct.agg(["mean", "sem"])
        ax.errorbar(s.index, s["mean"], yerr=s["sem"], marker="o", capsize=3, label=p)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K")
    ax.set_ylabel("relative bias in omega (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "fig_phase1_nonlinear", out_dir)


def fig_realdata(datasets, out_dir):
    """K=1 vs K=high, grouped bar chart per omega parameter, one panel per
    real dataset. `datasets` is a list of (name, df) pairs."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    for ax, (name, df) in zip(axes[0], datasets):
        om_cols = [c for c in df.columns if c.startswith("om_")]
        k_lo, k_hi = df.K.min(), df.K.max()
        lo = df[df.K == k_lo].iloc[0]
        hi = df[df.K == k_hi].iloc[0]
        x = np.arange(len(om_cols))
        width = 0.35
        ax.bar(x - width / 2, [lo[c] for c in om_cols], width, label=f"K={k_lo}")
        ax.bar(x + width / 2, [hi[c] for c in om_cols], width, label=f"K={k_hi}")
        ax.set_xticks(x)
        ax.set_xticklabels(om_cols)
        ax.set_ylabel("omega estimate")
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    savefig(fig, "fig_realdata_k1_vs_khigh", out_dir)


def fig_deltaofv(conditions, out_dir):
    """Empirical dOFV histogram vs the Self-Liang mixture reference density,
    one panel per (label, df) condition."""
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    for ax, (label, df) in zip(axes[0], conditions):
        dofv = df.dofv.values
        ax.hist(dofv, bins=30, density=True, alpha=0.6, label="empirical")
        xx = np.linspace(0.01, max(dofv.max(), 8), 300)
        ax.plot(xx, 0.5 * stats.chi2.pdf(xx, df=1), "r-", lw=2,
               label="0.5 x chi-sq(1)")
        ax.axvline(0, color="k", ls="--", alpha=0.5)
        ax.set_xlabel("dOFV")
        ax.set_ylabel("density")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    savefig(fig, "fig_deltaofv_calibration", out_dir)


def fig_psis(df, out_dir):
    """Per-subject ESS distribution, one panel per arm."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for arm, color in zip(sorted(df.arm.unique()), ["tab:blue", "tab:red"]):
        sub = df[df.arm == arm]
        ax.hist(sub.ess, bins=20, alpha=0.6, label=arm, color=color)
    ax.set_xlabel("effective sample size")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "fig_psis_ess", out_dir)


# %% ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-csv", default=None)
    ap.add_argument("--phase1-csv", default=None)
    ap.add_argument("--nonlinear-csv", default=None)
    ap.add_argument("--theoph-csv", default=None)
    ap.add_argument("--warfarin-csv", default=None)
    ap.add_argument("--deltaofv-free-csv", default=None)
    ap.add_argument("--deltaofv-amortized-csv", default=None)
    ap.add_argument("--psis-csv", default=None)
    ap.add_argument("--out", default="publication/figures")
    args = ap.parse_args()

    print("=" * 72)
    print("MANUSCRIPT FIGURES")
    print("=" * 72)

    df = try_load(args.phase0_csv, "Phase 0")
    if df is not None:
        fig_phase0(df, args.out)

    df = try_load(args.phase1_csv, "Phase 1 grid")
    if df is not None:
        fig_phase1_grid(df, args.out)

    df = try_load(args.nonlinear_csv or args.phase1_csv, "Phase 1 nonlinear")
    if df is not None and "nonlinear" in df.get("scenario", pd.Series(dtype=str)).unique():
        fig_nonlinear(df, args.out)
    elif df is not None:
        print("[skip] nonlinear figure: no 'nonlinear' scenario rows in the given file")

    realdata = []
    df = try_load(args.theoph_csv, "Theophylline")
    if df is not None:
        realdata.append(("theophylline", df))
    df = try_load(args.warfarin_csv, "warfarin")
    if df is not None:
        realdata.append(("warfarin", df))
    if realdata:
        fig_realdata(realdata, args.out)

    calib = []
    df = try_load(args.deltaofv_free_csv, "dOFV free")
    if df is not None:
        calib.append(("free", df))
    df = try_load(args.deltaofv_amortized_csv, "dOFV amortized")
    if df is not None:
        calib.append(("amortized", df))
    if calib:
        fig_deltaofv(calib, args.out)

    df = try_load(args.psis_csv, "PSIS")
    if df is not None:
        fig_psis(df, args.out)

    print(f"\nDone. Figures written to {args.out}/")
