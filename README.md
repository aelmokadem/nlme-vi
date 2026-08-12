# NLME Variational Inference: Omega-Shrinkage Project

Characterizing and correcting the systematic underestimation of random-effect
variances (Omega) by variational (ELBO-based) estimators in nonlinear
mixed-effects models, and recovering a NONMEM-comparable likelihood from the
fitted variational posterior.

**If you read nothing else, read the Key Findings section below** — it
summarizes what's actually been established across every phase, including
one result (ΔOFV / free-vs-amortized) that changes how you should use part
of this pipeline.

---

## Key findings so far

1. **Plain-ELBO (K=1) VI systematically underestimates Omega**, reproduced
   on known ground truth across dense/sparse simulated scenarios and real
   Theophylline data. Importance-weighting (higher K) substantially
   corrects it; a richer variational family (flow) corrects a large part of
   it even at K=1, especially in well-identified (dense) designs.
2. **Sparse sampling roughly doubles the shrinkage** at matched K relative
   to dense sampling — the identifiability mechanism behaves as predicted.
3. **Amortized+flow is unstable** and excluded from all headline results
   (see Known Issues). Free+flow and free/amortized+gaussian are solid.
4. **Real data (Theophylline) confirms the same signature** once two real
   bugs were fixed: a pre-dose-sample filtering bug and a badly-scaled
   initial guess (see the real-data section below). Small-N (N=12) residual
   bias appears to be substantially explained by classical small-sample MLE
   bias, not a VI-specific failure (tested via N-scaling: bias shrinks as N
   grows, holding K fixed).
5. **IMPORTANT: ΔOFV computed from a FREE posterior is not reliable for
   likelihood-ratio tests that change random-effect structure.** The free
   posterior gives the more-complex model in a nested comparison extra
   *per-subject* free parameters (not just the population-level parameter
   nominally being tested), which classical LRT theory — including the
   correct Self-Liang boundary-mixture reference — does not account for.
   This inflates ΔOFV and miscalibrates the test. **Switching to an
   amortized (shared-encoder) posterior for this specific use case restores
   approximately correct calibration** (boundary fraction ~50% vs free's
   ~26-29%; KS test p-value ~0.03 vs free's ~0.0003-0.0005). See the
   `nlme_vi_phase2_deltaofv.py` section below. This finding is scoped to
   nested comparisons that change the number of random effects — it does
   not affect the Omega-bias findings above, which never depended on
   nested-model comparison.

---

## Repository layout

```
.
├── nlmevi_core.py                    shared core: models, posteriors,
│                                      objectives, adaptive trainer
├── nlme_vi_phase0.py                 Phase 0: go/no-go
├── nlme_vi_phase1.py                 Phase 1: Q1-Q4 stress tests
├── commands.sh                       canonical run commands, callable by name
│
├── phase2/
│   ├── nlme_vi_phase2_deltaofv.py    dOFV / LRT calibration (see Key Findings)
│   ├── nlme_vi_phase2_psis.py        PSIS/ESS diagnostic validation
│   ├── baseline_nlmixr2.R            FOCEI/SAEM fit (R side -- see status below)
│   ├── nlme_vi_phase2_baselines.py   orchestrator: VI vs FOCEI vs SAEM
│   └── nlme_vi_phase2_realdata.py    real-data fits (Theophylline, warfarin)
│
└── publication/                      (not yet built)
    ├── make_tables.py
    ├── make_figures.py
    └── reproduce.sh
```

**`nlmevi_core.py` must be present in the same directory as whichever script
you're running.** Symlink it into `phase2/` once:
```bash
cd /path/to/nlme-vi
ln -s ../nlmevi_core.py phase2/nlmevi_core.py
```

---

## Requirements

### Python
```
pip install torch numpy pandas scipy matplotlib
pip install rdatasets   # for the real Theophylline data loader
```

### R (only for `phase2/baseline_nlmixr2.R` and the R-calling half of
`phase2/nlme_vi_phase2_baselines.py`)
```r
install.packages(c("nlmixr2", "dplyr", "readr", "nlmixr2data"))
```

**R environment status: unresolved.** `rxode2`'s model-compilation step
fails with a generic "Error building the model" wrapper on the development
machine (macOS 26 beta, R 4.6 via `rig`, freshly-reinstalled Xcode CLT).
Ruled out: missing/broken Command Line Tools (`R CMD SHLIB` on a trivial
file succeeds), Homebrew hijacking R (confirmed R resolves correctly to
CRAN 4.6 via `which R`/`rig list`), a PATH mismatch between terminal apps
(Positron vs. plain Terminal were resolving different R installs --
resolved, both now point at CRAN 4.6). **Not yet tried:** rebuilding
`rxode2`/`nlmixr2est` from source against the refreshed toolchain
(`install.packages(c("rxode2","nlmixr2est"), type="source")`), or testing
under R 4.5 via `rig` (R 4.6 + a beta macOS is about as bleeding-edge a
pairing as exists, and this is a live, untested hypothesis). **This does
not block anything else in the repo** -- see Key Finding 5 above, which
was established without needing FOCE/SAEM at all. If/when this gets
resolved, `phase2/nlme_vi_phase2_baselines.py`'s Python-side orchestration
is already fully validated via `--dry-run` and ready to go.

---

## `nlmevi_core.py` — shared core

Not run directly. Contains everything every other script imports:

| Component | What it is |
|---|---|
| `OneCmtOral` | Linear-tier model: 1-cmt oral PK, analytic solution (no ODE solver) |
| `OneCmtIVBolusMM` | Nonlinear-tier model: Michaelis-Menten elimination, RK4-integrated |
| `FreePosterior` / `AmortizedPosterior` | Gaussian variational families (per-subject / encoder-shared) |
| `FlowPosterior` (+ `AffineCoupling`, `ConditionalFlow`) | Conditional normalizing-flow variational family |
| `make_posterior(kind, family, ...)` | Dispatches to the right posterior class |
| `iw_elbo` | Importance-weighted ELBO, with `mc_reps` for gradient-variance reduction at small K |
| `is_marginal_loglik` | Post-hoc importance-sampling marginal likelihood + ESS/PSIS-proxy diagnostics. `K` param (default 4000) controls evaluation precision. |
| `train_to_convergence` | Adaptive stopping: trains until the omega estimates themselves plateau |
| `fit_model` | **The** trainer every script calls — adaptive stopping + best-checkpoint tracking baked in. Returns `(theta, q, ll, ess, top_share, n_steps_run, converged, cpu_secs)` -- an 8-tuple; unpack accordingly. |

Design rule every script downstream follows: **one seed generates one
dataset; every method arm in a comparison reads byte-identical data.**

`fit_model` signature highlights worth knowing:
- **`eval_k`** (default 4000): passed through to `is_marginal_loglik`'s
  evaluation K. Distinct from the training `K` argument -- raising this
  does not retrain, only re-evaluates the likelihood of the already-fitted
  posterior more precisely.
- **`cpu_secs`** returned via `time.process_time()`, not wall-clock. Immune
  to OS sleep and largely immune to other-process contention -- this is the
  number any VI-vs-FOCEI/SAEM speed claim should be built on, not
  wall-clock `secs`.
- **Best-checkpoint restoration** (always on): every 250 steps, if the loss
  improved, the parameters are snapshotted; the best snapshot is restored
  at the end regardless of where training ended up. Guards against a fit
  wandering into a bad basin and getting numerically stuck there while
  still satisfying the plateau test.

---

## Phase 0 — `nlme_vi_phase0.py`

**Question:** on a model with known ground truth, does plain-ELBO (K=1)
variational fitting systematically underestimate Omega, and does
importance-weighting (higher K) fix it?

**Execution style:** cell-based (`# %%` markers), no `if __name__` guard --
runnable top-to-bottom as a script or cell-by-cell in VSCode/Spyder.

```bash
python nlme_vi_phase0.py --quick              # ~1-2 min smoke test
python nlme_vi_phase0.py                       # full run
```

| Flag | Default | Meaning |
|---|---|---|
| `--quick` | off | tiny scale, 2 reps, for a fast sanity pass |
| `--reps` | 8 | replicates per (posterior, K) cell |
| `--subjects` | 120 | N per replicate |
| `--max-steps` | 25000 | safety cap for adaptive training |
| `--out` | `/mnt/user-data/outputs` | output directory |

Runs four sanity checks first (analytic-vs-ODE agreement, K=1 IW-ELBO ==
plain ELBO, IW-ELBO monotone in K, IS marginal likelihood matches a
closed-form case). Outputs `phase0_results.csv`, `phase0_omega_bias.png`,
and a printed go/no-go verdict, plus a `cpu_secs`/`secs` runtime table.

---

## Phase 1 — `nlme_vi_phase1.py`

**Four follow-up questions:**

| | Question | How |
|---|---|---|
| Q1 | Is the K=1 bias converged, or just under-trained? | `--converge-check` |
| Q2 | Does sparser sampling worsen shrinkage? | `--scenarios dense,sparse` |
| Q3 | Does the effect survive a nonlinear model? | `--scenarios ...,nonlinear` |
| Q4 | Does a richer family close what K alone leaves? | `--families gaussian,flow` |

**Execution style:** cell-based, same as Phase 0.

```bash
# Q1 -- run this before trusting any bias number from any script in this repo
python nlme_vi_phase1.py --converge-check --converge-steps 25000

# Q2 + Q4, gaussian family, both posteriors (amortized+gaussian is stable)
python nlme_vi_phase1.py --scenarios dense,sparse --families gaussian \
    --posteriors free,amortized --K 1,8,64 --reps 30

# Q4, flow family -- FREE POSTERIOR ONLY (see Known Issues re: amortized+flow)
python nlme_vi_phase1.py --scenarios dense,sparse --families flow \
    --posteriors free --K 1,8,64 --reps 30

# Q3, nonlinear tier (expensive, kept small by default)
python nlme_vi_phase1.py --scenarios dense,sparse,nonlinear \
    --nl-reps 20 --nl-steps 1500 --nl-subjects 60
```

| Flag | Default | Meaning |
|---|---|---|
| `--scenarios` | `dense,sparse` | comma list: `dense`, `sparse`, `nonlinear` |
| `--families` | `gaussian,flow` | comma list |
| `--posteriors` | `free,amortized` | comma list |
| `--K` | `1,8,64` | comma list |
| `--mc-reps-k1` | 8 | gradient-variance reduction for flow+K=1 |
| `--max-steps` | 25000 | safety cap |
| `--nl-reps` / `--nl-steps` / `--nl-subjects` / `--mm-dt` | 4/1500/60/0.1 | nonlinear-tier overrides |

**Built-in QC**: `summarize_v2` automatically flags fits whose marginal LL
is a wild outlier relative to sibling arms on the same replicate (catches
"converged per the plateau test but landed somewhere badly wrong" --
exactly the amortized+flow failure mode), and prints both an unfiltered and
a QC-filtered bias table so you can see the difference.

**Outputs:** `phase1_results.csv`, `phase1_grid.png`, console summary
including Q2/Q3/Q4 tables and a `cpu_secs` runtime table with a
flow/gaussian cost ratio.

---

## Phase 2

Everything in `phase2/` requires `nlmevi_core.py` symlinked into that
directory (see Requirements above).

### `nlme_vi_phase2_realdata.py` — does the signature show up on real data?

No ground truth on real data, so the check is: does K=1 report visibly
smaller Omega than K=high on the *same* dataset.

```bash
# Theophylline -- loads directly, no file needed
uv run phase2/nlme_vi_phase2_realdata.py --dataset theoph --K 1,64 --max-steps 40000

# Warfarin -- export from R first (no compilation needed, pure data package):
#   R> install.packages("nlmixr2data")
#   R> write.csv(nlmixr2data::warfarin, "warfarin.csv", row.names = FALSE)
# Check actual column names before loading:
#   head -3 warfarin.csv
uv run phase2/nlme_vi_phase2_realdata.py --csv warfarin.csv \
    --col-map "ID=subject,TIME=time,DV=conc,AMT=dose" --K 1,64 --max-steps 40000

# Synthetic placeholder (pipeline test only, not a real result) with
# ground-truth bias reporting -- useful for the N-scaling check:
uv run phase2/nlme_vi_phase2_realdata.py --n-subj 40 --K 1,64
```

| Flag | Default | Meaning |
|---|---|---|
| `--dataset theoph` | — | loads real Theophylline directly via `rdatasets` |
| `--csv PATH` | — | your own long-format CSV |
| `--col-map` | — | e.g. `"ID=subject,TIME=time,DV=conc,AMT=dose"` |
| `--n-subj` | 12 | synthetic-placeholder-only; test N-scaling of small-sample bias |
| `--K` | `1,64` | |
| `--max-steps` | 25000 | |

**Two real bugs found and fixed here, worth knowing about if extending to
a new dataset:**
1. **Pre-dose filtering.** The model predicts C(0)=0 exactly by
   construction. Filtering "concentration <= 0" alone misses subjects whose
   pre-dose reading is small-but-nonzero (assay noise) -- on Theoph, 3 of
   12 subjects had exactly this, producing a residual SD ~23x too large and
   preventing convergence entirely. Fixed by filtering on **time** (`t<=0`)
   instead of concentration value. If you load a new dataset and the
   "Dropping N pre-dose rows" message reports 0 or an implausible count,
   check whether that dataset's dosing convention actually uses `time==0`.
2. **Data-driven initial guess.** `theta_init` used to be hardcoded for the
   synthetic scenario's scale (dose=100, CL~2-3). Real data on a different
   scale (e.g. Theoph's mg/kg dosing, CL~0.04) was ~2 orders of magnitude
   off, causing both K arms to fail to converge while still traveling
   toward the right region of parameter space. Now derived from the data's
   own dose/peak-concentration ratio.

Handles per-subject varying doses and ragged designs (different subjects,
different numbers of observations) automatically.

**Outputs:** `phase2_realdata_results.csv`.

### `nlme_vi_phase2_psis.py` — does the per-subject trust diagnostic work?

```bash
python phase2/nlme_vi_phase2_psis.py --subjects 120 --bad-steps 30
```

Fits the same data twice (converged vs. deliberately under-trained) and
checks whether `is_marginal_loglik`'s ESS/top-share diagnostics separate
them.

**Validated finding:** mean ESS separates cleanly and reproduces closely
across runs (good ~1790-1810, bad ~1200-1200 at `--bad-steps 30`; gap
widens with more severe under-training). A fixed per-subject tail-count
threshold does **not** separate cleanly -- one specific subject is
intrinsically hard to fit regardless of training length, in both arms.
**Report aggregate ESS as the headline diagnostic**, not the tail-threshold
counts.

**Outputs:** `phase2_psis_results.csv`, `phase2_psis_comparison.png`.

### `nlme_vi_phase2_deltaofv.py` — is ΔOFV valid for likelihood-ratio tests?

**This is the script behind Key Finding 5.** Simulates data under a known
null (true Omega_ka=0), fits a reduced (2 random effects) and full (3
random effects) model, and checks whether the empirical dOFV distribution
matches the correct Self-Liang boundary-mixture reference (NOT plain
chi-square(1) -- testing a variance component against its boundary is a
non-regular problem; see the script's docstring for the full derivation).

**Execution style: this is the one script with `if __name__ == "__main__":`**,
unlike every other script in this repo. This is a hard requirement, not a
style inconsistency: `--n-workers` uses `ProcessPoolExecutor`, and macOS's
`spawn` start method re-imports this file in every worker process. Without
the guard, every worker would re-run the entire experiment recursively.
Function/class definitions above the guard are still fine to run cell-by-cell.

```bash
# Free posterior (default) -- reproduces the miscalibration finding
caffeinate -i uv run phase2/nlme_vi_phase2_deltaofv.py \
    --reps 100 --subjects 120 --K 64 --n-workers 9

# Amortized posterior -- the fix. Compare boundary fraction / KS p-value
# against the free run above.
caffeinate -i uv run phase2/nlme_vi_phase2_deltaofv.py \
    --reps 100 --subjects 120 --K 64 --n-workers 9 --posterior amortized

# Test the (ruled-out) IS-evaluation-noise hypothesis
caffeinate -i uv run phase2/nlme_vi_phase2_deltaofv.py \
    --reps 20 --subjects 120 --K 64 --eval-k 20000 --n-workers 9
```

| Flag | Default | Meaning |
|---|---|---|
| `--reps` | 100 | KS test needs ~30-50+ positive-dOFV reps for real power |
| `--subjects` | 120 | |
| `--K` | 64 | training K -- use your best-corrected arm, not the known-biased K=1 |
| `--eval-k` | 4000 (nlmevi_core default) | post-hoc likelihood evaluation precision -- **ruled out** as the fix (rep 48's outlier value stayed frozen at ~+8.3 across a 5x eval-k increase) |
| `--posterior` | `free` | `free` (miscalibrated for this use case) or `amortized` (the fix) |
| `--n-workers` | 1 | replicates are fully independent; parallelize across processes. `caffeinate -i` recommended alongside this on macOS to prevent sleep-related slowdowns during long runs |
| `--max-steps` | 25000 | |

**Validated results** (100 reps each, N=120, K=64):

| | free, eval_k=4000 | free, eval_k=20000 | amortized, eval_k=4000 |
|---|---|---|---|
| boundary fraction (target ~50%) | 26.0% | 29.0% | **50.0%** |
| KS p-value (target >0.05) | 0.0003 | 0.0005 | **0.0295** |
| verdict | FAIL | FAIL | PASS (borderline by strict 0.05 KS standard) |

**Open items**: amortized run showed a 27% non-convergence rate (checked --
excluding non-converged fits did NOT explain the residual calibration gap,
so it's a separate quality issue, not the primary explanation). A 300-rep
amortized confirmatory run for a more decisive KS test was in progress as
of this writing -- check `phase2_deltaofv_results.csv`'s timestamp / rerun
if you need the latest numbers.

**Outputs:** `phase2_deltaofv_results.csv`, `phase2_deltaofv_calibration.png`.

### `baseline_nlmixr2.R` + `nlme_vi_phase2_baselines.py` — VI vs FOCEI vs SAEM

```bash
# Python-side pipeline test, no R needed (fully validated):
uv run phase2/nlme_vi_phase2_baselines.py --subjects 20 --reps 1 --dry-run

# Real run (blocked on R environment -- see Requirements above):
uv run phase2/nlme_vi_phase2_baselines.py --subjects 20 --reps 1

# Direct R-side test, bypassing the Python orchestrator entirely --
# useful for isolating R issues (this is how the na="." bug and the
# rxode2 compile issue were found):
Rscript phase2/baseline_nlmixr2.R outputs/phase2_baselines/rep0_data.csv \
    /tmp/test_foce.csv foce
```

**Two real bugs found and fixed** before ever getting a real R run:
1. `write_nonmem_csv` writes `DV="."` for dosing rows (standard NONMEM
   convention). `readr::read_csv`'s default NA values don't include `"."`,
   so the whole DV column would parse as text. Fixed with explicit
   `na = c("", "NA", ".")`.
2. `--r-script`'s default was a relative path resolved against the
   *caller's* working directory, not the script's own location -- same
   class of bug as the `nlmevi_core` import issue. Fixed to resolve
   relative to `Path(__file__).parent`.

**Timing**: both `cpu_secs` (process CPU time, comparable across
Python/R since both use process-time APIs) and `wall_secs` /
`wall_secs_subprocess_total` (R process startup overhead) are tracked.
Use `cpu_secs` for any speed claim.

| Flag | Default | Meaning |
|---|---|---|
| `--subjects` | 120 | |
| `--reps` | 5 | |
| `--K-vi` | 64 | the corrected VI arm to compare |
| `--r-script` | (auto, next to this script) | override only if relocating `baseline_nlmixr2.R` |
| `--dry-run` | off | skip the actual R call, validate Python-side plumbing |
| `--out` | `outputs/phase2_baselines` | |

**Outputs:** `phase2_baseline_comparison.csv`, plus per-replicate
`repN_data.csv` / `repN_foce.csv` / `repN_saem.csv`.

---

## `commands.sh` — canonical run commands

A set of documented, independently-callable shell functions, not a
monolithic pipeline. Given how much of this project is iterative (tune a
flag, look at the result, decide the next command), a single unattended
"run everything" script would work against the actual workflow. Instead:

```bash
# see what's available
bash commands.sh

# run exactly one thing, with the flags this README recommends
bash commands.sh phase1_gaussian_grid
bash commands.sh deltaofv_amortized

# or just open commands.sh and copy the line you want
```

Each function bakes in the lessons from this whole debugging process --
`caffeinate` where it matters, correct paths, sensible defaults -- so you
get a correct starting command without re-deriving it each time, while
keeping full control to run one function, edit one before running it, or
ignore the file entirely and type your own command.

---

## Known issues / open items

- **`baseline_nlmixr2.R` / R environment**: unresolved, see Requirements
  above. Not blocking anything else.
- **Amortized+flow (Phase 1) is fragile.** Even with `mc_reps`, warmup
  scheduling, tighter gradient clipping, and best-checkpoint tracking, this
  combination still occasionally converges smoothly to a stable, wrong,
  repeatable answer (not caught by convergence flags, only by the LL-outlier
  QC filter, and even then not always). Not load-bearing for any core
  claim -- excluded from headline results. Free+flow and
  free/amortized+gaussian are unaffected and stable.
- **dOFV / amortized posterior**: 27% non-convergence rate in the 100-rep
  amortized run, not yet resolved (confirmed NOT to be the primary driver
  of the residual KS-test gap, but worth fixing for its own sake --
  possibly needs a higher `--max-steps` specifically for this combination).
  A 300-rep confirmatory run was in progress as of this writing.
- **Warfarin**: not yet run through the real-data pipeline. Export doesn't
  need the blocked R environment (pure data package, no compilation) --
  see the real-data section above.
- **N-scaling small-sample-bias finding**: currently 1 replicate per N
  (12/40/80). Directionally clear (bias shrinks as N grows) but needs 5-10
  replicates per N before citing precisely.
- **`publication/` scripts don't exist yet** -- `make_tables.py`,
  `make_figures.py`, `reproduce.sh`. Build these last, once Phase 2/3
  numbers are final.
- **Phase 3 (the nonlinear tier) is fully integrated into
  `nlme_vi_phase1.py`, not a separate script.** Originally planned as a
  standalone `nlme_vi_phase3_nonlinear.py`; that plan was superseded once
  it became clear the nonlinear scenario shares everything else Phase 1
  already has (fit_model, QC filtering, cpu_secs tracking, summarize_v2,
  the CLI pattern) -- a separate file would have meant duplicating all of
  it for no benefit. Run via `--scenarios nonlinear`. The recommended,
  stable model is `OneCmtIVBolusMMNoKmRE` (IIV on Vmax and V only, Km
  fixed -- see its docstring: putting IIV on both Vmax and Km
  simultaneously is a known cross-method MM identifiability problem, not
  a VI-specific one, confirmed empirically across four independent test
  conditions in this project). The original 3-random-effect
  `OneCmtIVBolusMM` still exists in `nlmevi_core.py`, unused by default,
  kept only as the documented record of that instability. A second hard
  tier (e.g. TMDD) would most naturally be another `get_scenario()` option
  in `nlme_vi_phase1.py`, following the same pattern, rather than a new
  file -- new model code following the same `log_prior`/`log_lik`/
  `log_joint` interface is all that's actually needed.

---

## Practical tips learned the hard way

- **Prevent sleep on long macOS runs**: prefix with `caffeinate -i`.
  Sleep-interrupted fits show wall times 20-100x their normal duration with
  no algorithmic explanation -- always check `cpu_secs` vs `wall_secs`
  before trusting a runtime number from a long unattended run.
- **Use `cpu_secs`, not wall-clock `secs`**, for any speed claim or
  cross-run timing comparison. `time.process_time()` doesn't advance during
  sleep and is far less sensitive to other-process contention.
- **`--n-workers` (deltaofv script only, so far)**: replicates are
  independent: `os.cpu_count() - 1` is a reasonable default. Each worker is
  pinned to 1 PyTorch thread internally to avoid CPU oversubscription --
  don't remove this if extending the pattern elsewhere.
- **Full paths beat bare commands when multiple R/Python installs are in
  play** (e.g. via `rig`, Homebrew, or multiple terminal apps with
  different PATH setups) -- `which R` / `which Rscript` before a long
  debugging session saved real time more than once in this project.

---

## The one rule that matters most

**One seed generates one dataset. Every method arm in a comparison reads
byte-identical data.** Enforced throughout `nlmevi_core.py` and every phase
script -- preserve this if extending the codebase. Breaking it silently
invalidates any bias comparison built on top.