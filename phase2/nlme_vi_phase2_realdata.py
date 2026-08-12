"""
Phase 2 / real data: does the shrinkage signature show up outside simulation?
================================================================================

Every result so far is on simulated data with known ground truth. This
script applies the same VI machinery to a REAL oral-PK dataset -- but
because there's no ground truth on real data, the check here is different
from Phase 0/1: instead of bias-vs-truth, we look for the SAME SIGNATURE
found on simulated data: K=1 (plain ELBO) reporting visibly smaller
between-subject variability than K=high (IW-ELBO), on the identical dataset.
That's the pattern a reviewer familiar with Tarek & Afonso's warfarin table
will recognize immediately.

DATA SOURCES

    Theophylline: loaded directly, no CSV needed --
        pip install rdatasets
        python nlme_vi_phase2_realdata.py --dataset theoph
    This is the actual Boeckmann/Sheiner/Beal data (verified: 132 rows, 12
    subjects, matches the canonical dataset) via the `rdatasets` package,
    which mirrors R's datasets programmatically -- not retyped from memory,
    so no transcription-error risk.

    Warfarin: NOT available through rdatasets or any other Python package
    checked (it lives in the specialized `nlmixr2data` R package, which
    general-purpose R-dataset mirrors don't index). Export it from R:
        write.csv(nlmixr2data::warfarin, "warfarin.csv", row.names = FALSE)
    then point --csv at it with --col-map to match column names (see below)
    -- no need to hand-edit the file or this script.

    Any other CSV: --csv plus --col-map.

EXPECTED INTERNAL FORMAT (after loading/renaming, long format):
    subject, time, conc, dose
    Different subjects may have different numbers of observations (ragged
    designs are handled via padding + mask) and different doses.

--col-map lets you point --csv at a file with different column names
without editing anything, e.g. for a raw nlmixr2data::warfarin export
(columns typically ID, TIME, DV, AMT or similar -- check your file and
adjust):
    --col-map "ID=subject,TIME=time,DV=conc,AMT=dose"

REQUIRES  nlmevi_core.py in the same directory.
"""

import argparse

import numpy as np
import pandas as pd
import torch
import math

from nlmevi_core import OneCmtOral, fit_model

torch.set_default_dtype(torch.float64)


# %% --------------------------------------------------------- data loading
def load_theoph():
    """
    Loads the real Boeckmann/Sheiner/Beal theophylline dataset via the
    `rdatasets` package (pip install rdatasets) -- programmatically sourced,
    not retyped, so no transcription-error risk. Verified against the
    canonical dataset: 132 rows, 12 subjects.
    """
    try:
        import rdatasets
    except ImportError:
        raise ImportError(
            "pip install rdatasets   (then re-run with --dataset theoph)"
        )
    df = rdatasets.data("datasets", "Theoph")
    return df.rename(columns={"Subject": "subject", "Time": "time", "Dose": "dose"})
    # 'conc' column name already matches; 'Wt' and 'rownames' are simply
    # ignored downstream since load_real_data only reads the 4 required columns


def apply_col_map(df, col_map_str):
    """--col-map 'ID=subject,TIME=time,DV=conc,AMT=dose' -> renames those
    columns in place. Silently ignores mappings for columns not present."""
    if not col_map_str:
        return df
    mapping = {}
    for pair in col_map_str.split(","):
        src, dst = pair.split("=")
        mapping[src.strip()] = dst.strip()
    return df.rename(columns=mapping)


def load_real_data(df):
    """
    Takes a dataframe already containing (subject, time, conc, dose) columns
    -- from load_theoph(), a --csv + --col-map, or hand-built -- and pads to
    a common (N, T_max) shape with a mask for ragged designs, since real
    data rarely has every subject sampled at the same times.

    Returns (data_dict, dose_per_subject, subject_ids) where data_dict has
    the same 'logy'/'t'/'mask' keys nlmevi_core.to_tensors expects.
    """
    required = {"subject", "time", "conc", "dose"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {missing}. "
                         f"Found: {list(df.columns)}. Use --col-map to rename "
                         f"columns from your source file without editing it.")

    # Extract per-subject dose from the ORIGINAL, UNFILTERED data, using
    # max() rather than "first row after filtering." This matters because
    # two real, common dose-column conventions behave completely
    # differently once the t=0 filtering below runs:
    #   (a) Theoph-style: dose is a repeated per-subject covariate, same
    #       value on every row (including observation rows). Any surviving
    #       row after filtering has the right value.
    #   (b) NONMEM/warfarin-style: dose only appears (as AMT) on the
    #       EVID=1 dosing row itself; every observation row has amt=0.
    #       Since the t=0 filter below DROPS the dosing row, "first row
    #       after filtering" silently returns 0 for every subject under
    #       this convention -- dose collapses to 0, every predicted
    #       concentration collapses to exactly 0, and the fit fails in a
    #       new, undiagnosed way (found when loading real warfarin data,
    #       which uses convention (b)). max() over the subject's rows
    #       BEFORE any filtering is correct under both conventions: for
    #       (a) it just returns the repeated value; for (b) it correctly
    #       picks up the one nonzero (dosing-row) value regardless of
    #       whether that row survives later filtering.
    dose_lookup = df.groupby("subject").dose.max()

    # Drop t=0 (dosing-time) observations. This is a real bug fix, not a
    # cosmetic one: the model predicts C(0)=0 EXACTLY by construction (no
    # absorption has occurred at the instant of dosing, for ANY subject
    # parameters -- individual etas cannot change this). Filtering on
    # "concentration <= 0" alone MISSES pre-dose samples that happen to be
    # small-but-nonzero (assay noise near the LLOQ, or an early-absorption
    # artifact) -- on real Theophylline data, 3 of 12 subjects have exactly
    # this: nonzero t=0 readings of 0.15-0.74 that survived a
    # concentration-based filter, and diverging by ~25+ log-units from an
    # unavoidable model prediction of ~0 at that instant. A handful of
    # points like that dominate the single shared sigma parameter
    # completely and produce a residual SD so large (~4 on the log scale,
    # i.e. ~55x typical multiplicative error) that NEITHER K=1 nor K=64 can
    # converge to anything meaningful -- the failure looks like a training
    # problem but is actually a fixed, unfittable structural mismatch at
    # one time point per subject. Filtering by TIME (the actual criterion
    # for "this is a pre-dose sample") instead of by concentration value
    # fixes this regardless of what number happened to be recorded there.
    n_before = len(df)
    t0 = df[df.time <= 1e-6]
    if len(t0) > 0:
        print(f"  Dropping {len(t0)} pre-dose (t=0) rows "
              f"(subjects: {sorted(t0.subject.unique().tolist())}, "
              f"concentrations: {sorted(t0.conc.round(2).tolist())}) -- the model "
              f"predicts C(0)=0 by construction regardless of concentration filtering, "
              f"so these are dropped by TIME, not by value.")
        df = df[df.time > 1e-6].copy()

    # Separate, orthogonal safety net for any remaining non-positive
    # concentrations elsewhere in a trajectory (post-absorption BLQ, not a
    # pre-dose artifact). NOT appropriate for genuine BLQ censoring during
    # the elimination phase -- that needs proper M3/M4 handling, not
    # implemented here. Printed explicitly so it's never a silent decision.
    bad = df[df.conc <= 0]
    if len(bad) > 0:
        print(f"  Dropping {len(bad)} additional non-positive concentration rows "
              f"(subjects: {sorted(bad.subject.unique().tolist())}) -- "
              f"if these are BLQ censoring rather than artifacts, this simple "
              f"drop is NOT appropriate; implement M3/M4 instead.")
        df = df[df.conc > 0].copy()
    print(f"  {len(df)}/{n_before} rows retained")

    subjects = df.subject.unique()
    n_subj = len(subjects)
    n_obs_max = df.groupby("subject").size().max()

    logy = np.full((n_subj, n_obs_max), np.nan)
    t = np.zeros((n_subj, n_obs_max))
    mask = np.zeros((n_subj, n_obs_max))
    dose = np.zeros(n_subj)

    for i, sid in enumerate(subjects):
        sub = df[df.subject == sid].sort_values("time")
        n = len(sub)
        conc = sub.conc.values.astype(float)
        logy[i, :n] = np.log(conc)
        t[i, :n] = sub.time.values
        mask[i, :n] = 1.0
        dose[i] = dose_lookup.loc[sid]

    if dose_lookup.min() <= 0:
        zero_dose_subjects = dose_lookup[dose_lookup <= 0].index.tolist()
        raise ValueError(
            f"Subjects with dose <= 0 after max()-based extraction: "
            f"{zero_dose_subjects}. This means NO row for that subject "
            f"(in the entire original file, before any filtering) had a "
            f"positive value in the mapped dose column -- almost always a "
            f"--col-map mistake (wrong source column) or a dataset that "
            f"encodes dose somewhere this loader doesn't check (e.g. dose "
            f"only in a header/metadata row, or a multi-dose regimen where "
            f"'dose' should mean something other than a single scalar)."
        )

    if len(np.unique(dose)) > 1:
        print(f"  NOTE: doses vary across subjects ({sorted(np.unique(dose))}). "
              f"Handled correctly: OneCmtOral's arithmetic broadcasts fine "
              f"against a per-subject dose tensor (verified: doubling one "
              f"subject's dose exactly doubles their predicted concentration "
              f"while leaving others unchanged) -- run_real_data_comparison "
              f"below passes dose as a (N,1) tensor rather than a scalar.")

    logy = np.nan_to_num(logy, nan=0.0)   # masked positions; value unused
    data = dict(logy=logy, t=t, mask=mask)
    return data, dose, subjects


def make_synthetic_placeholder(n_subj=12, seed=0):
    """
    NOT REAL DATA. Generates a small synthetic oral-PK dataframe purely so
    this script's loading/fitting pipeline can be exercised without a real
    dataset on hand. Use --dataset theoph or --csv for any actual result.

    Unlike real data, the ground truth is known here -- returns it alongside
    the dataframe so the caller can report actual bias, not just the K=1 vs
    K=high relative comparison that's all that's available for real data.
    This matters specifically for small n_subj: it lets you distinguish the
    VI-specific K-dependent shrinkage this whole project studies from
    classical small-sample MLE bias in variance components (well known,
    unrelated to VI -- it's why REML exists), which VI's K correction cannot
    be expected to fix because it isn't the same mechanism.
    """
    from nlmevi_core import TrueParams, simulate
    tp = TrueParams()
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
    d = simulate(tp, n_subj, times, seed=seed)
    rows = []
    for i in range(n_subj):
        for j, tt in enumerate(times):
            rows.append(dict(subject=i + 1, time=tt,
                            conc=math.exp(d["logy"][i, j]), dose=tp.dose))
    print(f"  [placeholder] using SYNTHETIC (not real) data, n_subj={n_subj}")
    truth = dict(om_CL=tp.om_CL, om_V=tp.om_V, om_ka=tp.om_ka)
    return pd.DataFrame(rows), truth


def estimate_init_theta(df, dose):
    """
    Data-driven initial guess, replacing a previously-hardcoded theta_init
    that was tuned for the synthetic placeholder scenario (dose=100,
    CL~2-3, V~20-30). That guess was silently ~2 orders of magnitude off
    for real data on a different scale -- e.g. Theoph's mg/kg dosing gives
    CL~0.04, V~0.5. The consequence wasn't just a slower fit: both K=1 and
    K=64 spent their entire step budget traveling toward the right region
    of parameter space and were caught mid-journey at different points,
    producing a large, spurious K=1-vs-K=64 "difference" that was really
    just two different snapshots of an incomplete, still-converging
    trajectory -- not a real K-dependent signature.

    Rough moment-based guess, not a proper method-of-moments PK estimate:
        V0  ~ typical dose / typical peak concentration
        CL0 ~ V0 * a generic elimination rate (assumes an ~7h half-life,
              reasonable across many oral-PK compounds as a STARTING point;
              the fit still estimates the real value, this only needs to be
              in the right order of magnitude, not correct)
        ka0 = 1.0 (generic; typical value estimation corrects it)
    """
    peak_conc = max(df.groupby("subject").conc.max().median(), 1e-6)
    typical_dose = df.groupby("subject").dose.first().median()
    V0 = max(typical_dose / peak_conc, 1e-3)
    CL0 = V0 * 0.1
    ka0 = 1.0
    print(f"  data-driven init: CL0={CL0:.4g} V0={V0:.4g} ka0={ka0:.4g} "
         f"(from typical dose={typical_dose:.4g}, peak conc={peak_conc:.4g})")
    return [math.log(CL0), math.log(V0), math.log(ka0),
           math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]


# %% ------------------------------------------------------------ experiment
def run_real_data_comparison(df, K_grid, max_steps, out):
    data, dose, subjects = load_real_data(df)
    n_subj = len(subjects)
    print(f"Loaded {n_subj} subjects")

    # Per-subject dose tensor, shape (N,1) -- OneCmtOral's arithmetic
    # broadcasts against this exactly like it does against CL/V/ka, so
    # varying real-world doses (e.g. Theoph's weight-adjusted mg/kg dosing)
    # are handled correctly without restricting to a single-dose subset.
    dose_tensor = torch.as_tensor(dose, dtype=torch.float64).unsqueeze(-1)
    model = OneCmtOral(dose_tensor)
    theta_init = estimate_init_theta(df, dose)

    rows = []
    for K in K_grid:
        theta, q, ll, ess, top, n_run, converged, _ = fit_model(
            model, data, "free", "gaussian", K, theta_init, n_eta=3,
            max_steps=max_steps,
        )
        est = np.concatenate([np.exp(theta[:3]), np.exp(theta[3:6]), [np.exp(theta[6])]])
        rows.append(dict(K=K, CL=est[0], V=est[1], ka=est[2],
                        om_CL=est[3], om_V=est[4], om_ka=est[5], sigma=est[6],
                        ofv=-2 * ll, mean_ess=float(ess.mean()),
                        n_steps=n_run, converged=converged))
        flag = "" if converged else "  *** DID NOT CONVERGE ***"
        print(f"  K={K:3d} | om_CL {est[3]:.3f} om_V {est[4]:.3f} om_ka {est[5]:.3f} | "
              f"OFV {-2*ll:8.1f} | steps={n_run}{flag}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{out}/phase2_realdata_results.csv", index=False)
    return results_df


def summarize(df, truth=None):
    print("\n" + "=" * 72)
    print("ALL PARAMETERS, both K arms (fixed effects + omegas + sigma)")
    print("=" * 72)
    param_cols = ["CL", "V", "ka", "om_CL", "om_V", "om_ka", "sigma"]
    tab = df.set_index("K")[param_cols].T
    print(tab.round(4).to_string())
    print("\n(the fixed effects -- CL, V, ka -- are expected to stay roughly stable")
    print("across K; large swings there, unlike the omegas, would be a red flag,")
    print("not the expected signature.)")

    print("\n" + "-" * 72)
    print("K=1 (plain ELBO) vs K=high (IW-ELBO): the signature to look for")
    print("-" * 72)
    k_lo, k_hi = df.K.min(), df.K.max()
    lo = df[df.K == k_lo].iloc[0]
    hi = df[df.K == k_hi].iloc[0]
    for p in ["om_CL", "om_V", "om_ka"]:
        rel = 100 * (lo[p] - hi[p]) / hi[p]
        print(f"  {p}: K={k_lo} -> {lo[p]:.3f}   K={k_hi} -> {hi[p]:.3f}   "
              f"(K={k_lo} is {rel:+.1f}% relative to K={k_hi})")
    print(f"\n  OFV: K={k_lo} -> {lo.ofv:.1f}   K={k_hi} -> {hi.ofv:.1f}")
    print("\nIf K=1's omegas are consistently smaller than K=high's (negative %")
    print("above), that's the same shrinkage signature found on simulated data,")
    print("now on real data where FOCE/SAEM (via phase2_baselines.py) is the")
    print("external reference to compare both arms against.")

    if truth is not None:
        print("\n" + "-" * 72)
        print("GROUND TRUTH COMPARISON (only possible because this is synthetic --")
        print("real data has no equivalent check). If K=high still shows large bias")
        print("here, that's NOT explained by the K-dependent VI mechanism this project")
        print("studies -- at N this small, classical MLE small-sample bias in variance")
        print("components (the reason REML exists) is a live, separate hypothesis, and")
        print("would affect FOCE/SAEM too if they're also run with ML rather than REML.")
        for p in ["om_CL", "om_V", "om_ka"]:
            for row, k in [(lo, k_lo), (hi, k_hi)]:
                bias = 100 * (row[p] - truth[p]) / truth[p]
                print(f"  {p} @ K={k:<3d}: est={row[p]:.3f}  truth={truth[p]:.3f}  "
                     f"bias={bias:+.1f}%")


# %% ------------------------------------------------------------- config
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["theoph"], default=None,
                help="load a real dataset directly (no CSV needed); "
                     "requires: pip install rdatasets")
ap.add_argument("--csv", default=None,
                help="path to a real long-format CSV. Use --col-map if "
                     "its columns aren't already named subject,time,conc,dose "
                     "(e.g. a raw nlmixr2data::warfarin export).")
ap.add_argument("--col-map", default=None,
                help="e.g. 'ID=subject,TIME=time,DV=conc,AMT=dose'")
ap.add_argument("--K", default="1,64")
ap.add_argument("--n-subj", type=int, default=12,
                help="subjects for the SYNTHETIC placeholder only (ignored for "
                     "--dataset/--csv real data). Use this to test whether "
                     "residual K=high bias shrinks as N grows -- if it does, "
                     "that's evidence of classical small-sample MLE bias, not "
                     "a failure of the K-dependent VI correction.")
ap.add_argument("--max-steps", type=int, default=25000)
ap.add_argument("--out", default="outputs")
args, _ = ap.parse_known_args()

K_GRID = [int(k) for k in args.K.split(",")]
MAX_STEPS, OUT = args.max_steps, args.out
print(f"config: dataset={args.dataset}  csv={args.csv}  K_GRID={K_GRID}")


# %% ------------------------------------------------------------ load data
truth = None
if args.dataset == "theoph":
    print("Loading the real Theophylline dataset via rdatasets...")
    data_df = load_theoph()
elif args.csv:
    data_df = apply_col_map(pd.read_csv(args.csv), args.col_map)
else:
    print("*** No --dataset or --csv given: generating a SYNTHETIC placeholder. ***")
    print("*** This validates the pipeline only -- NOT a real-data result. ***")
    print("*** Use --dataset theoph for real data, or --csv for your own. ***\n")
    data_df, truth = make_synthetic_placeholder(n_subj=args.n_subj)
# `data_df` is now in your session -- inspect it directly, e.g.
# data_df.groupby('subject').size() to check the design.


# %% ------------------------------------------------------------------ fit
df = run_real_data_comparison(data_df, K_GRID, MAX_STEPS, OUT)


# %% -------------------------------------------------------- summarize
summarize(df, truth=truth)
print(f"\nRaw results -> {OUT}/phase2_realdata_results.csv")
