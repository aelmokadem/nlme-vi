"""
Phase 2 / baselines: VI vs. FOCEI vs. SAEM, on the same simulated data
========================================================================

Nothing in Phase 0/1 was ever checked against real FOCE/SAEM output -- every
comparison so far has been VI talking to itself (different K, different
family, different posterior). This script closes that gap: it writes the
exact same simulated dataset both to nlmevi_core's own tensors (for VI) and
to a NONMEM-format CSV (for nlmixr2's FOCEI/SAEM via baseline_nlmixr2.R),
fits all three, and reports them side by side against known ground truth.

*** THE R SIDE OF THIS IS UNVALIDATED IN THIS SANDBOX -- see the warning at
the top of baseline_nlmixr2.R. Use --dry-run first to check the Python-side
CSV format and orchestration logic without needing R installed; that part
IS fully tested. ***

REQUIRES  nlmevi_core.py in the same directory, and (for real runs) R with
nlmixr2 installed and baseline_nlmixr2.R alongside this script.
"""

import argparse
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nlmevi_core import TrueParams, OneCmtOral, simulate, fit_model


# %% --------------------------------------------------- data format bridge
def write_nonmem_csv(data, dose, times, path):
    """
    Converts a nlmevi_core-style data dict (logy, t, mask -- numpy arrays,
    subjects x observations) into a NONMEM/nlmixr2-format long CSV:
    one dosing row (TIME=0, AMT=dose, EVID=1) per subject, followed by one
    observation row per (subject, time) with DV = log(concentration) --
    matching OneCmtOral's generative model, where the residual error is
    additive on the log scale.
    """
    n_subj, n_obs = data["logy"].shape
    rows = []
    for i in range(n_subj):
        subj_id = i + 1  # nlmixr2/NONMEM convention: 1-indexed
        rows.append(dict(ID=subj_id, TIME=0.0, DV=".", AMT=dose, EVID=1, CMT=1))
        for j in range(n_obs):
            rows.append(dict(
                ID=subj_id, TIME=float(times[j]), DV=float(data["logy"][i, j]),
                AMT=0, EVID=0, CMT=2,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def run_r_baseline(input_csv, output_csv, method, r_script, dry_run=False):
    """
    Calls baseline_nlmixr2.R via subprocess. In --dry-run mode, skips the
    actual R call and writes a placeholder (NaN) result row instead, so the
    rest of the pipeline (merging, comparison table) can be exercised and
    tested without R installed.
    """
    if dry_run:
        pd.DataFrame([dict(
            method=method, CL=np.nan, V=np.nan, ka=np.nan,
            om_CL=np.nan, om_V=np.nan, om_ka=np.nan, sigma=np.nan,
            ofv=np.nan, logLik=np.nan, cpu_secs=np.nan, wall_secs=np.nan,
            n_subjects=np.nan,
        )]).to_csv(output_csv, index=False)
        return pd.read_csv(output_csv)

    t0 = time.time()
    result = subprocess.run(
        ["Rscript", str(r_script), str(input_csv), str(output_csv), method],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        print(f"  *** Rscript ({method}) FAILED (exit {result.returncode}) ***")
        print("  --- stdout ---"); print(result.stdout[-3000:])
        print("  --- stderr ---"); print(result.stderr[-3000:])
        raise RuntimeError(f"nlmixr2 {method} fit failed -- see output above")
    print(f"  Rscript ({method}) completed in {time.time()-t0:.1f}s")
    return pd.read_csv(output_csv)


# %% ------------------------------------------------------------ experiment
def run_comparison(n_subj, times, n_reps, out_dir, r_script, dry_run=False,
                   K_vi=64, max_steps=25000, seed0=6000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tp = TrueParams()
    truth = dict(CL=tp.CL, V=tp.V, ka=tp.ka, om_CL=tp.om_CL, om_V=tp.om_V,
                om_ka=tp.om_ka, sigma=tp.sigma)
    model = OneCmtOral(tp.dose)
    theta_init = [math.log(2.0), math.log(20.0), math.log(0.8),
                 math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.3)]

    rows = []
    for rep in range(n_reps):
        seed = seed0 + rep
        print(f"\n--- replicate {rep} (seed {seed}) ---")
        data = simulate(tp, n_subj, times, seed=seed)   # ONE dataset, all three methods

        # ---- VI (K=1: known-biased arm, and K=K_vi: corrected arm) ----
        for K, tag in [(1, "vi_K1"), (K_vi, f"vi_K{K_vi}")]:
            t0 = time.time()
            theta, q, ll, ess, top, n_run, converged, cpu_secs = fit_model(
                model, data, "free", "gaussian", K, theta_init, n_eta=3,
                max_steps=max_steps,
            )
            wall_secs = time.time() - t0
            est = np.concatenate([np.exp(theta[:3]), np.exp(theta[3:6]), [np.exp(theta[6])]])
            rows.append(dict(replicate=rep, method=tag,
                            CL=est[0], V=est[1], ka=est[2],
                            om_CL=est[3], om_V=est[4], om_ka=est[5], sigma=est[6],
                            ofv=-2 * ll, converged=converged, n_steps=n_run,
                            cpu_secs=cpu_secs, wall_secs=wall_secs,
                            wall_secs_subprocess_total=np.nan))
            print(f"  {tag:10s} | om_CL {est[3]:.3f} om_V {est[4]:.3f} om_ka {est[5]:.3f} "
                 f"| OFV {-2*ll:8.1f} | converged={converged} | "
                 f"cpu {cpu_secs:.1f}s / wall {wall_secs:.1f}s")

        # ---- FOCEI / SAEM via nlmixr2 ----
        csv_in = out_dir / f"rep{rep}_data.csv"
        write_nonmem_csv(data, tp.dose, times, csv_in)
        for method in ["foce", "saem"]:
            csv_out = out_dir / f"rep{rep}_{method}.csv"
            t0 = time.time()
            try:
                r_res = run_r_baseline(csv_in, csv_out, method, r_script, dry_run=dry_run)
                subprocess_wall_secs = time.time() - t0
                r = r_res.iloc[0]
                # cpu_secs = the R script's own proc.time()-based fit-only CPU
                # time -- the fair apples-to-apples column against VI's
                # fit_model cpu_secs (both use process CPU time, immune to
                # sleep/contention, neither counts interpreter startup).
                # wall_secs = R's own wall-clock fit time (context only).
                # wall_secs_subprocess_total additionally includes R process
                # startup + package loading -- real cost of this subprocess
                # design, not part of the algorithmic speed comparison.
                r_cpu_secs = getattr(r, "cpu_secs", np.nan)
                r_wall_secs = getattr(r, "wall_secs", np.nan)
                rows.append(dict(replicate=rep, method=method,
                                CL=r.CL, V=r.V, ka=r.ka, om_CL=r.om_CL, om_V=r.om_V,
                                om_ka=r.om_ka, sigma=r.sigma, ofv=r.ofv,
                                converged=np.nan, n_steps=np.nan,
                                cpu_secs=r_cpu_secs, wall_secs=r_wall_secs,
                                wall_secs_subprocess_total=subprocess_wall_secs))
                print(f"  {method:10s} | om_CL {r.om_CL} om_V {r.om_V} om_ka {r.om_ka} "
                     f"| OFV {r.ofv} | cpu {r_cpu_secs}s / wall {r_wall_secs}s / "
                     f"subprocess {subprocess_wall_secs:.1f}s")
            except Exception as e:
                print(f"  {method:10s} | FAILED: {e}")
                rows.append(dict(replicate=rep, method=method,
                                CL=np.nan, V=np.nan, ka=np.nan, om_CL=np.nan,
                                om_V=np.nan, om_ka=np.nan, sigma=np.nan, ofv=np.nan,
                                converged=np.nan, n_steps=np.nan,
                                cpu_secs=np.nan, wall_secs=np.nan,
                                wall_secs_subprocess_total=time.time() - t0))

    df = pd.DataFrame(rows)
    for name, val in truth.items():
        df[f"truth_{name}"] = val
    return df


def summarize(df):
    print("\n" + "=" * 78)
    print("MEAN ESTIMATE BY METHOD (vs. ground truth)")
    print("=" * 78)
    cols = ["CL", "V", "ka", "om_CL", "om_V", "om_ka", "sigma"]
    tab = df.groupby("method")[cols].mean()
    truth_row = df[[f"truth_{c}" for c in cols]].iloc[0]
    truth_row.index = cols
    tab.loc["TRUTH"] = truth_row
    print(tab.round(3).to_string())

    print("\n" + "-" * 78)
    print("Mean OFV by method (lower is not directly comparable across VI-K vs FOCE/SAEM")
    print("unless the underlying likelihoods are on the same scale -- check this before")
    print("over-interpreting OFV differences here):")
    print(df.groupby("method").ofv.mean().round(1).to_string())

    print("\n" + "-" * 78)
    print("RUNTIME -- the number a VI-vs-FOCEI/SAEM speed claim should rest on.")
    print("  cpu_secs  = process CPU time (Python time.process_time() / R proc.time()'s")
    print("              user.self+sys.self on each side). Immune to OS sleep and largely")
    print("              immune to other-process contention -- USE THIS for the speed claim.")
    print("  wall_secs = fit-only wall-clock, for context; can be inflated by anything that")
    print("              paused/slowed the machine (a >1.5x gap vs cpu_secs is a sign the")
    print("              wall_secs number for that run is contaminated, not a real cost).")
    print("  wall_secs_subprocess_total = FOCE/SAEM only; ALSO includes R process startup +")
    print("              package loading -- real cost of the current subprocess-per-fit")
    print("              design, not an algorithmic cost.")
    rt = df.groupby("method")[["cpu_secs", "wall_secs", "wall_secs_subprocess_total"]].agg(
        ["mean", "median"])
    print(rt.round(2).to_string())

    vi_cols = [c for c in df.method.unique() if c.startswith("vi_")]
    for vi_col in vi_cols:
        vi_cpu = df[df.method == vi_col].cpu_secs.mean()
        for baseline in ["foce", "saem"]:
            base_cpu = df[df.method == baseline].cpu_secs.mean()
            if pd.notna(vi_cpu) and pd.notna(base_cpu) and vi_cpu > 0:
                print(f"  {baseline.upper()} / {vi_col} CPU-time speedup: {base_cpu/vi_cpu:.1f}x")

    print("\n  CAVEAT: cpu_secs removes sleep/contention noise but NOT thread-count")
    print("  differences -- pin thread counts on both sides before quoting a speed")
    print("  number in the paper (torch.set_num_threads(N) for VI; check nlmixr2/your")
    print("  BLAS's thread settings for FOCE/SAEM), or the comparison still conflates")
    print("  the algorithm with incidental hardware parallelism.")


# %% ------------------------------------------------------------- config
ap = argparse.ArgumentParser()
ap.add_argument("--subjects", type=int, default=120)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--K-vi", type=int, default=64)
ap.add_argument("--max-steps", type=int, default=25000)
ap.add_argument("--r-script", default=str(Path(__file__).parent / "baseline_nlmixr2.R"),
                help="defaults to baseline_nlmixr2.R next to this script, "
                     "regardless of what directory you run this from")
ap.add_argument("--dry-run", action="store_true",
                help="skip the actual Rscript call; validate the Python-side "
                     "pipeline (CSV format, merging, summary) without R installed")
ap.add_argument("--out", default="outputs/phase2_baselines")
args, _ = ap.parse_known_args()

N_SUBJ, N_REPS, K_VI = args.subjects, args.reps, args.K_vi
MAX_STEPS, R_SCRIPT, DRY_RUN, OUT = args.max_steps, args.r_script, args.dry_run, args.out
TIMES = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
print(f"config: N_SUBJ={N_SUBJ}  N_REPS={N_REPS}  K_VI={K_VI}  DRY_RUN={DRY_RUN}")


# %% --------------------------------------------------------- experiment
print("=" * 78)
print(f"VI vs FOCEI vs SAEM  N={N_SUBJ}  reps={N_REPS}  dry_run={DRY_RUN}")
if DRY_RUN:
    print("*** DRY RUN: R is not being called. FOCE/SAEM rows will be NaN. ***")
print("=" * 78)

df = run_comparison(N_SUBJ, TIMES, N_REPS, OUT, R_SCRIPT,
                    dry_run=DRY_RUN, K_vi=K_VI, max_steps=MAX_STEPS)
df.to_csv(f"{OUT}/phase2_baseline_comparison.csv", index=False)
# `df` (VI + FOCE + SAEM estimates, one row per method per replicate) is
# now in your interactive session.


# %% -------------------------------------------------------- summarize
summarize(df)
print(f"\nRaw results -> {OUT}/phase2_baseline_comparison.csv")
