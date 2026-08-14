"""
publication/make_tables.py -- manuscript tables from the project's result CSVs.

WHY EXPLICIT FILE PATHS FOR EVERYTHING, NOT DEFAULT FILENAMES:
    Several scripts write to a fixed default output filename regardless of
    which condition was run (e.g. nlme_vi_phase2_realdata.py always writes
    phase2_realdata_results.csv whether you ran --dataset theoph or
    warfarin; nlme_vi_phase2_deltaofv.py always writes
    phase2_deltaofv_results.csv whether --posterior was free or amortized).
    Running a second condition after a first SILENTLY OVERWRITES the first
    file. This script does not guess -- it takes an explicit path per named
    condition, so if you ran multiple conditions you must have already
    saved/renamed each one (e.g. `cp phase2_realdata_results.csv
    phase2_realdata_theoph.csv` immediately after the Theoph run, before
    running warfarin). If a path is not supplied or the file doesn't exist,
    that table is skipped with a clear note -- never silently produces a
    table from the wrong condition's data.

USAGE
    python publication/make_tables.py \
        --phase0-csv outputs/phase0_results.csv \
        --phase1-csv outputs/phase1_results.csv \
        --nonlinear-csv outputs/phase1_nonlinear_results.csv \
        --theoph-csv outputs/phase2_realdata_theoph.csv \
        --warfarin-csv outputs/phase2_realdata_warfarin.csv \
        --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
        --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized.csv \
        --psis-csv outputs/phase2_psis_results.csv \
        --baseline-csv outputs/phase2_baseline_comparison.csv \
        --out publication/tables

Any flag can be omitted; that table is skipped, not an error. Run with
whatever you have -- tables fill in as more results become available.

OUTPUT: for each table, writes both a .csv (for import into stats software
/ further processing) and a .md (for direct copy-paste into a manuscript
draft) to --out, plus prints every table to console.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


# %% --------------------------------------------------------- utilities
def df_to_markdown(df, index=False):
    """
    Minimal manual markdown-table writer -- avoids depending on the
    `tabulate` package (which pandas.DataFrame.to_markdown requires but
    doesn't ship with), so this script has no extra dependencies beyond
    what the rest of the project already needs.
    """
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def save_table(df, name, out_dir, index=True):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    md_path = os.path.join(out_dir, f"{name}.md")
    df.to_csv(csv_path, index=index)
    with open(md_path, "w") as f:
        f.write(df_to_markdown(df, index=index))
    print(f"\n=== {name} ===")
    print(df.to_string(index=index))
    print(f"-> {csv_path}\n-> {md_path}")


def try_load(path, label):
    if not path:
        print(f"\n[skip] {label}: no path given")
        return None
    if not os.path.exists(path):
        print(f"\n[skip] {label}: {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"\n[loaded] {label}: {path} ({len(df)} rows)")
    return df


# %% ------------------------------------------------------ table builders
def table_phase0(df):
    """Headline go/no-go: Omega bias by posterior x K x param.

    Grouped by param (not just posterior x K): pooling om_CL/om_V/om_ka
    together would mix qualitatively different parameters into one
    mean_bias_pct/sd_bias_pct, inflating sd_bias_pct with cross-parameter
    variation rather than reflecting replicate-to-replicate uncertainty
    for a single parameter -- the fixed_effects companion table below was
    already correctly grouped this way; this was the one place that
    wasn't, now fixed to match."""
    om = df[df.param.str.startswith("om_")]
    t = om.groupby(["posterior", "K", "param"]).rel_bias_pct.agg(["mean", "std", "count"])
    t.columns = ["mean_bias_pct", "sd_bias_pct", "n_estimates"]
    return t.round(2).reset_index()


def table_phase0_fixed_effects(df):
    """Validation companion to table_phase0: fixed-effects (CL/V/ka/sigma)
    bias by posterior x K. Expected to stay near-flat/near-zero across K,
    UNLIKE the omegas -- this is the check that the shrinkage phenomenon is
    specific to variance components, not a general estimation problem."""
    fe = df[~df.param.str.startswith("om_")]
    t = fe.groupby(["posterior", "K", "param"]).rel_bias_pct.agg(["mean", "std", "count"])
    t.columns = ["mean_bias_pct", "sd_bias_pct", "n_estimates"]
    return t.round(2).reset_index()


def table_phase1_grid(df):
    """Q2/Q4: Omega bias by scenario x posterior x family x K x param,
    linear scenarios only (dense/sparse) -- nonlinear handled separately
    since it has different parameter names and is typically run
    standalone. Grouped by param for the same reason as table_phase0
    above -- see its docstring."""
    om = df[(df.param.str.startswith("om_")) & (df.scenario.isin(["dense", "sparse"]))]
    t = om.groupby(["scenario", "posterior", "family", "K", "param"]).rel_bias_pct.agg(
        ["mean", "std", "count"])
    t.columns = ["mean_bias_pct", "sd_bias_pct", "n_estimates"]
    return t.round(2).reset_index()


def table_phase1_grid_fixed_effects(df):
    """Validation companion to table_phase1_grid: same grouping, fixed
    effects instead of omegas. Same purpose as table_phase0_fixed_effects,
    across the full Q2/Q4 grid rather than just Phase 0's single scenario."""
    fe = df[(~df.param.str.startswith("om_")) & (df.scenario.isin(["dense", "sparse"]))]
    t = fe.groupby(["scenario", "posterior", "family", "K", "param"]).rel_bias_pct.agg(
        ["mean", "std", "count"])
    t.columns = ["mean_bias_pct", "sd_bias_pct", "n_estimates"]
    return t.round(2).reset_index()


def table_nonlinear(df):
    """Q3: nonlinear (MM) tier -- all parameters, not just omegas, since
    fixed-effect recovery (Vmax/Km/V) is itself part of the reported
    finding, not just the omega shrinkage signature."""
    nl = df[df.scenario == "nonlinear"] if "scenario" in df.columns else df
    t = nl.groupby(["posterior", "family", "K", "param"]).agg(
        mean_estimate=("estimate", "mean"), truth=("truth", "first"),
        mean_bias_pct=("rel_bias_pct", "mean"), n=("rel_bias_pct", "count"),
        frac_converged=("converged", "mean"),
    )
    return t.round(3).reset_index()


def table_realdata(df, dataset_name):
    """K=1 vs K=high, all parameters, for one real dataset."""
    k_lo, k_hi = df.K.min(), df.K.max()
    lo = df[df.K == k_lo].iloc[0]
    hi = df[df.K == k_hi].iloc[0]
    param_cols = [c for c in ["CL", "V", "ka", "om_CL", "om_V", "om_ka", "sigma"]
                 if c in df.columns]
    rows = []
    for p in param_cols:
        rel = 100 * (lo[p] - hi[p]) / hi[p] if hi[p] != 0 else np.nan
        rows.append(dict(dataset=dataset_name, param=p,
                        K_low=k_lo, estimate_K_low=lo[p],
                        K_high=k_hi, estimate_K_high=hi[p],
                        pct_diff_low_vs_high=round(rel, 2) if pd.notna(rel) else np.nan))
    return pd.DataFrame(rows)


def table_deltaofv_calibration(df, label):
    """Boundary fraction + KS test vs the Self-Liang mixture reference, for
    one (posterior, eval_k) condition."""
    dofv = df.dofv.values
    frac_boundary = 100 * np.mean(dofv <= 0.05)
    positive = dofv[dofv > 0.05]
    if len(positive) >= 10:
        ks_stat, ks_p = stats.kstest(positive, "chi2", args=(1,))
    else:
        ks_stat, ks_p = np.nan, np.nan
    correct_cutoff = stats.chi2.ppf(0.90, df=1)
    type1_correct = 100 * np.mean(dofv > correct_cutoff)
    n_negative = int((dofv < 0).sum())
    return pd.DataFrame([dict(
        condition=label, n_reps=len(dofv),
        boundary_fraction_pct=round(frac_boundary, 1),
        n_negative_dofv=n_negative,
        ks_stat=round(ks_stat, 4) if pd.notna(ks_stat) else np.nan,
        ks_p=round(ks_p, 4) if pd.notna(ks_p) else np.nan,
        type1_error_correct_test_pct=round(type1_correct, 1),
    )])


def table_psis(df):
    """Aggregate ESS by arm -- the headline diagnostic (see README: the
    per-subject tail-threshold metric does NOT separate cleanly and is
    intentionally not reported as a summary table here)."""
    t = df.groupby("arm").ess.agg(["mean", "median", "min", "max"])
    return t.round(1).reset_index()


def table_baseline_comparison(df):
    """VI vs FOCEI vs SAEM, all parameters + runtime. Only meaningful once
    the R side is unblocked; skipped entirely if not supplied."""
    cols = ["CL", "V", "ka", "om_CL", "om_V", "om_ka", "sigma"]
    cols = [c for c in cols if c in df.columns]
    est = df.groupby("method")[cols].mean()
    truth_cols = [f"truth_{c}" for c in cols if f"truth_{c}" in df.columns]
    if truth_cols:
        truth_row = df[truth_cols].iloc[0]
        truth_row.index = [c.replace("truth_", "") for c in truth_cols]
        est.loc["TRUTH"] = truth_row
    if "cpu_secs" in df.columns:
        rt = df.groupby("method").cpu_secs.agg(["mean", "median"])
        rt.columns = ["cpu_secs_mean", "cpu_secs_median"]
        est = est.join(rt)
    return est.round(4).reset_index()


# %% ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-csv", default=None)
    ap.add_argument("--phase1-csv", default=None,
                    help="dense/sparse grid (nonlinear rows, if present, "
                         "are excluded from the linear table automatically)")
    ap.add_argument("--nonlinear-csv", default=None,
                    help="nonlinear (MM) tier results -- can be the same "
                         "file as --phase1-csv if run together, or separate "
                         "if run standalone (recommended, given cost)")
    ap.add_argument("--theoph-csv", default=None)
    ap.add_argument("--warfarin-csv", default=None)
    ap.add_argument("--deltaofv-free-csv", default=None)
    ap.add_argument("--deltaofv-amortized-csv", default=None)
    ap.add_argument("--psis-csv", default=None)
    ap.add_argument("--baseline-csv", default=None,
                    help="VI vs FOCEI vs SAEM -- only meaningful once the "
                         "R environment is working; skipped if not given")
    ap.add_argument("--out", default="publication/tables")
    args = ap.parse_args()

    print("=" * 72)
    print("MANUSCRIPT TABLES")
    print("=" * 72)

    df = try_load(args.phase0_csv, "Phase 0 (go/no-go)")
    if df is not None:
        save_table(table_phase0(df), "table_phase0_headline", args.out)
        save_table(table_phase0_fixed_effects(df), "table_phase0_fixed_effects", args.out)

    df = try_load(args.phase1_csv, "Phase 1 grid (Q2/Q4, linear)")
    if df is not None:
        save_table(table_phase1_grid(df), "table_phase1_q2q4", args.out)
        save_table(table_phase1_grid_fixed_effects(df), "table_phase1_q2q4_fixed_effects", args.out)

    df = try_load(args.nonlinear_csv or args.phase1_csv, "Phase 1 nonlinear (Q3)")
    if df is not None and "nonlinear" in df.get("scenario", pd.Series(dtype=str)).unique():
        save_table(table_nonlinear(df), "table_phase1_q3_nonlinear", args.out)
    elif df is not None:
        print("[skip] Q3 nonlinear table: no 'nonlinear' scenario rows in the given file")

    realdata_tables = []
    df = try_load(args.theoph_csv, "Real data: Theophylline")
    if df is not None:
        realdata_tables.append(table_realdata(df, "theophylline"))
    df = try_load(args.warfarin_csv, "Real data: warfarin")
    if df is not None:
        realdata_tables.append(table_realdata(df, "warfarin"))
    if realdata_tables:
        save_table(pd.concat(realdata_tables, ignore_index=True),
                  "table_realdata_k1_vs_khigh", args.out, index=False)

    calib_tables = []
    df = try_load(args.deltaofv_free_csv, "dOFV calibration: free posterior")
    if df is not None:
        calib_tables.append(table_deltaofv_calibration(df, "free"))
    df = try_load(args.deltaofv_amortized_csv, "dOFV calibration: amortized posterior")
    if df is not None:
        calib_tables.append(table_deltaofv_calibration(df, "amortized"))
    if calib_tables:
        save_table(pd.concat(calib_tables, ignore_index=True),
                  "table_deltaofv_calibration", args.out, index=False)

    df = try_load(args.psis_csv, "PSIS/ESS diagnostic")
    if df is not None:
        save_table(table_psis(df), "table_psis_ess", args.out, index=False)

    df = try_load(args.baseline_csv, "VI vs FOCEI vs SAEM")
    if df is not None:
        save_table(table_baseline_comparison(df), "table_baseline_comparison",
                  args.out, index=False)

    print("\n" + "=" * 72)
    print(f"Done. Tables written to {args.out}/ (skipped tables just mean that")
    print("source wasn't supplied or wasn't found -- rerun with more flags")
    print("as more results become available.)")