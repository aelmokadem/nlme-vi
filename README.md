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
   data (Theophylline, warfarin). Importance-weighting (higher K)
   substantially corrects it; a richer variational family (flow) corrects a
   large part of it even at K=1, especially in well-identified (dense)
   designs.
2. **Sparse sampling roughly doubles the shrinkage** at matched K relative
   to dense sampling — the identifiability mechanism behaves as predicted.
3. **Amortized+flow is unstable at every tested K (1, 8, 64), not just at
   K=1** — QC-flagged suspect-fit rates are comparable across all three
   (8/7/6 flagged fits respectively in the production grid), and the single
   worst failure in the entire grid (LL ≈ -784,500) occurred at K=64, not
   K=1. Higher K does not rescue this combination; it changes the *shape*
   of the failure (from occasional loud divergence to more frequent
   convergence onto a stable, wrong, repeatable answer) without reducing
   its frequency. **This is specifically a flow-family problem, not a
   general amortized-posterior problem** — the amortized encoder's output
   serves double duty as both the flow's base distribution and its
   conditioning input, creating a moving-target optimization compounded by
   representational redundancy between the two components. Amortized
   posteriors under the plain Gaussian family show none of this (see
   Finding 5) and are excluded from nothing. Amortized+flow is excluded
   from all headline results at every K — see Known Issues.
4. **Real data confirms the same signature on two independent datasets.**
   Theophylline required fixing two real bugs first (a pre-dose-sample
   filtering bug and a badly-scaled initial guess); warfarin required
   fixing a third (dose extraction assumed a repeated-per-row covariate
   column, which broke on warfarin's NONMEM-standard convention of dose
   only appearing on the EVID=1 row) -- see the real-data section below.
   Once fixed, both datasets show K=1 under-reporting Omega relative to
   K=64, matching the simulated-data pattern (Theoph: Omega_V -18.2%;
   warfarin: Omega_V -8.6%, Omega_ka -17.0%, Omega_CL flat, consistent
   with CL being the most robustly-identified parameter throughout this
   project). Warfarin additionally gives an external validation point: the
   fitted fixed effects (CL~0.13 L/h, V~8.0 L) fall inside the 95%
   confidence intervals of nlmixr2's own published FOCEi fit on the same
   data (CL: 0.134 [0.125-0.142]; V: 7.96 [7.61-8.34]), despite this
   project's simpler 1-compartment structural model versus nlmixr2's
   two-step transit-absorption model -- real agreement on the parameters
   that should agree, not just a directional match. Theophylline's
   small-N (N=12) residual Omega bias is substantially explained by
   classical small-sample MLE bias rather than a VI-specific failure
   (tested via N-scaling: bias shrinks as N grows, holding K fixed).
5. **dOFV computed from a FREE posterior is not reliable for
   likelihood-ratio tests that change random-effect structure.** The free
   posterior gives the more-complex model in a nested comparison extra
   *per-subject* free parameters (not just the population-level parameter
   nominally being tested), which classical LRT theory — including the
   correct Self-Liang boundary-mixture reference — does not account for.
   This inflates dOFV and miscalibrates the test (boundary fraction ~26-29%
   against a target of ~50%; KS test p~0.0003-0.0005, decisively rejecting
   the reference distribution at both n=100 and n=300).
   **Switching to an amortized-GAUSSIAN posterior** (no flow -- see Finding
   3 for why that combination is excluded regardless) **for this specific
   use case corrects the boundary-mass component of the miscalibration**:
   boundary fraction reaches ~48-50%, reproducible at both n=100 and n=300.
   Confirmed as a real, converged-fit effect, not an artifact of
   non-convergence or insufficient post-hoc likelihood evaluation precision
   (both tested and ruled out as the explanation). **One caveat kept
   deliberately unresolved**: a smaller residual deviation in the shape of
   the non-boundary dOFV distribution persists under amortized-gaussian and
   is detectable with enough replicates (n=300, KS p<0.0001 -- the n=100
   "borderline pass" did not survive more data). A follow-up check found a
   real correlation between this residual and how much the reduced/full
   fits' sigma estimates diverge on the same data -- but sigma and the
   likelihood are mechanically coupled by construction (sigma appears
   directly inside the likelihood formula), so this correlation is more
   likely a byproduct of the same overfitting event the boundary-mixture
   theory already characterizes than a second, independent, previously-
   unaccounted-for mechanism. Treat the boundary-mass correction as
   established and the precise LRT p-value as still approximate; the source
   of the residual gap remains genuinely open. See the
   `nlme_vi_phase2_deltaofv.py` section below. This finding is scoped to
   nested comparisons that change the number of random effects — it does
   not affect the Omega-bias findings above, which never depended on
   nested-model comparison.
6. **The nonlinear (Michaelis-Menten) tier requires IIV restricted to
   Vmax and V only, with Km as a pure fixed effect** — not a workaround,
   but standard pharmacometric practice for MM models (Vmax/Km IIV
   correlation is a known cross-method identifiability problem, not
   VI-specific) and directly supported by this project's own data: across
   four independent conditions (two sampling designs, two step budgets),
   Omega_Vmax's bias stayed in a reasonable -10% to -16% range every time
   while Omega_Km's never recovered from -77% to -93%, regardless of
   training duration or design changes -- consistent with a genuine
   structural instability, not undertraining or a fixable design flaw.
   With Km's IIV removed (`OneCmtIVBolusMMNoKmRE`), fixed effects recover
   cleanly and the same K=1-vs-K=64 shrinkage-and-correction signature seen
   everywhere else in this project reproduces on Vmax and V's omegas.
   **Confirmed at full production scale (N=60, 20 reps, K=1/8/64):
   free+gaussian is clean and complete** -- Omega bias -1.79% (K=1) ->
   -0.47% (K=8) -> -0.04% (K=64), closing almost entirely, even better
   than the N=15 exploratory runs suggested (consistent with the
   small-sample-bias mechanism established elsewhere in this project: the
   correction gets cleaner as N grows). **This is the reportable
   nonlinear-tier headline result.**
   **Separately, and unexpectedly: amortized+gaussian failed
   catastrophically on this tier** -- bias +230% (K=1) -> +374% (K=8) ->
   +1168% (K=64), *worsening* with more K rather than correcting (26/120
   fits non-converged, 33/120 QC-outlier-flagged, some fixed-effect
   estimates reaching numerically absurd values, e.g. Km ~ 2*10^7). This
   is a NEW, previously uncharacterized failure, distinct from
   amortized+flow's known instability (Finding 3) -- no flow is involved
   here at all, so that mechanism doesn't apply. Leading (untested)
   hypothesis: the amortized encoder must learn one shared mapping from
   raw trajectory shape to (mu, log_s) across all subjects at once; this
   tier's design deliberately spans saturated/transition/first-order MM
   regimes (required to identify Vmax -- see above), so subject
   trajectories vary far more in shape than on the linear tier, which may
   be a harder generalization problem for a shared encoder than the free
   posterior's independently-optimized-per-subject approach. Not
   investigated further -- free+gaussian already fully answers Q3.
   **Important process note**: amortized+gaussian was never validated at
   small scale on this tier before the production run -- every prior
   exploratory check used `--posteriors free` only. This is a real gap in
   following this project's own established discipline of testing cheap
   before committing to an expensive run, worth remembering for any future
   tier/combination.

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
└── publication/
    ├── make_tables.py                 DONE -- manuscript tables from result CSVs
    ├── make_figures.py                DONE -- manuscript figures, no captions baked in
    └── reproduce.sh                   not yet built
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
| `set_device(device_str)` | Resolves `"cpu"`/`"auto"`/`"cuda"`/`"mps"` and updates the module-level `DEVICE` everything else reads at call time. **`mps` (Apple Silicon GPU) raises immediately, on purpose**: MPS does not support float64, a permanent Metal backend limitation, and this project requires float64 throughout (importance weights at high K span many orders of magnitude; float32 would silently corrupt the K=64 arm the corrected results depend on). CPU is the only supported option on Apple Silicon for this codebase -- confirmed by hitting the failure directly, not assumed. |

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

**Execution style: this script now ALSO uses `if __name__ == "__main__":`**,
same hard requirement and reason as `nlme_vi_phase2_deltaofv.py` (not a
style choice): `--n-workers > 1` uses `ProcessPoolExecutor`, and macOS's
`spawn` start method re-imports this file in every worker process --
without the guard, every worker would re-parse argv and re-run the entire
grid recursively. Function/class definitions above the guard are still
fine to run cell-by-cell; only the final execution block needs a full
script invocation.

```bash
# Q1 -- run this before trusting any bias number from any script in this repo
python nlme_vi_phase1.py --converge-check --converge-steps 25000

# Q2 + Q4, gaussian family, both posteriors (amortized+gaussian is stable)
python nlme_vi_phase1.py --scenarios dense,sparse --families gaussian \
    --posteriors free,amortized --K 1,8,64 --reps 30

# Q4, flow family -- FREE POSTERIOR ONLY (see Known Issues re: amortized+flow)
python nlme_vi_phase1.py --scenarios dense,sparse --families flow \
    --posteriors free --K 1,8,64 --reps 30

# Q3, nonlinear tier -- production scale, parallelized (see cost estimate
# below; expect several hours even with parallelism)
caffeinate -i python nlme_vi_phase1.py --scenarios nonlinear --families gaussian \
    --posteriors free,amortized --nl-reps 20 --nl-subjects 60 --mm-dt 0.1 \
    --n-workers 9 --out .
```

| Flag | Default | Meaning |
|---|---|---|
| `--scenarios` | `dense,sparse` | comma list: `dense`, `sparse`, `nonlinear` |
| `--families` | `gaussian,flow` | comma list |
| `--posteriors` | `free,amortized` | comma list |
| `--K` | `1,8,64` | comma list |
| `--device` | `cpu` | `cpu`/`auto`/`cuda`/`mps` -- `mps` raises immediately (see core-module table above), `cpu` is the only supported option on Apple Silicon |
| `--n-workers` | 1 | fits are fully independent (each rebuilds its own model+data from a seed); run this many in parallel processes. Try `os.cpu_count()-1`. Default 1 = original sequential behavior, unchanged. Only affects the Q2-4 grid, not `--converge-check`. |
| `--mc-reps-k1` | 8 | gradient-variance reduction for flow+K=1 |
| `--max-steps` | 25000 | safety cap |
| `--nl-reps` / `--nl-steps` / `--nl-subjects` / `--mm-dt` | 4/1500/60/0.1 | nonlinear-tier overrides |

**Built-in QC**: `summarize_v2` automatically flags fits whose marginal LL
is a wild outlier relative to sibling arms on the same replicate (catches
"converged per the plateau test but landed somewhere badly wrong" --
exactly the amortized+flow failure mode), and prints both an unfiltered and
a QC-filtered bias table so you can see the difference. A separate "Fixed
effects + sigma" table is also printed alongside the omega table --
generic across scenarios (works whether the scenario's parameter names are
`CL/V/ka`, as in dense/sparse, or `Vmax/Km/V`, as in nonlinear), since
fixed-effect bias was previously computed but not actually shown in the
console output for anything other than Phase 0.

**Nonlinear tier specifics** (see Key Finding 6 for the full story): the
default model is `OneCmtIVBolusMMNoKmRE` (IIV on Vmax and V only, Km
fixed) at dose=300 with a 60h observation window -- both settings are
load-bearing, not arbitrary, and are baked into `get_scenario()` rather
than exposed as flags, since a shorter/cheaper design was tested and found
to break Km's identifiability (not just undertrained -- genuinely
uninformative data in that regime). **Cost estimate for the production
command above**, computed from real measured timing (119 CPU-sec per 9000
steps at N=15/dt=0.5, scaled to N=60/dt=0.1): roughly 5-24 hours with
9-way parallelism depending on how many steps convergence actually takes
at N=60 (the one genuine unknown -- launch with `caffeinate` and check
back rather than trying to predict it more precisely).

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
3. **Dose extraction assumed the wrong convention.** Found when loading
   warfarin: Theoph's `Dose` column is a repeated per-subject covariate
   (same value on every row); warfarin follows the more common
   NONMEM/AMT convention -- dose is nonzero *only* on the EVID=1 dosing
   row, zero on every observation row. Extracting dose via "first
   surviving row after the pre-dose filter" (fine under Theoph's
   convention) silently returned 0 for every subject under warfarin's,
   since the pre-dose filter drops the one row carrying the real value.
   Fixed by taking `max()` over each subject's dose column on the
   *original, unfiltered* data -- correct under both conventions
   regardless of which row survives later filtering.

**Actual results obtained** (see Key Finding 4): Theoph (N=12, both K
arms converged) shows Omega_V -18.2% at K=1 vs K=64; warfarin (N=32, both
arms converged) shows Omega_V -8.6% and Omega_ka -17.0%, with fitted
CL/V falling inside nlmixr2's published 95% CI on the same data.

**CSV-overwrite gotcha**: this script always writes to the same default
filename (`phase2_realdata_results.csv`) regardless of which
`--dataset`/`--csv` was used. Running Theoph then warfarin without saving
between runs means the second silently overwrites the first. **Copy the
CSV to a dataset-specific name immediately after each run** (e.g. `cp
phase2_realdata_results.csv phase2_realdata_theoph.csv`) -- this is what
`publication/make_tables.py`/`make_figures.py` expect as input.

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
| `--posterior` | `free` | `free` (miscalibrated for this use case) or `amortized` (fixes the boundary-mass component -- see caveat below) |
| `--n-workers` | 1 | replicates are fully independent; parallelize across processes. `caffeinate -i` recommended alongside this on macOS to prevent sleep-related slowdowns during long runs |
| `--max-steps` | 25000 | |

**Validated results** (N=120, K=64):

| | free, n=100 | amortized, n=100 | amortized, n=300 (confirmatory) |
|---|---|---|---|
| boundary fraction (target ~50%) | 26.0% | **50.0%** | 48.3% |
| KS p-value (target >0.05) | 0.0003 | 0.0295 (borderline) | **<0.0001** |
| KS statistic (N-independent; compare directly) | ~0.20-0.24 | 0.2015 | **0.1864** |
| verdict | FAIL | PASS (borderline) | technically FAIL, but see below |

The n=300 KS p-value looks worse than n=100's, but this is a statistical
power effect, not evidence the fix failed -- the KS *statistic* (which,
unlike the p-value, doesn't shrink with sample size) barely moved
(0.2015 -> 0.1864) and stayed below every free-posterior run's statistic.
**The boundary-mass fraction (the part of the effective-degrees-of-freedom
mechanism this fix directly targets) is confirmed real and stable across
both sample sizes** (50.0% / 48.3%, both near the 50% target); what n=300
revealed is a smaller, separate residual deviation in the shape of the
non-boundary distribution, not a failure of the boundary-mass correction.
Excluding non-converged fits does NOT explain the residual gap (checked:
converged-only subset was, if anything, slightly further from the target).

**Sigma-divergence follow-up** (n=100, amortized): added `sigma_reduced`/
`sigma_full`/`sigma_diff` columns to test whether the reduced/full fits'
residual-error estimates diverging on the same data explains the residual
gap. Found a real correlation (`corr(sigma_diff, |dOFV|) = -0.536`, top-decile
|dOFV| replicates show ~2.5x larger |sigma_diff|) -- but sigma appears
directly inside the likelihood formula, so sigma and dOFV are mechanically
coupled by construction; an overfitting event on a given dataset would
naturally show up in both simultaneously, without sigma being an
independent, previously-unaccounted-for degree of freedom the way the
free posterior's per-subject parameters were. **Treat this as a real
correlation without a confirmed causal mechanism, not a second finding**
-- the residual gap's actual source remains open.

**Open items**: amortized runs showed a 23-27% non-convergence rate
across both n=100 and n=300 (confirmed NOT the primary driver of the
residual KS gap, but a separate quality issue worth its own fix -- likely
needs a higher `--max-steps` specifically for this combination, not yet
tried).

**Outputs:** `phase2_deltaofv_results.csv`, `phase2_deltaofv_calibration.png`.
Same overwrite gotcha as the real-data script: the default filename
doesn't encode which `--posterior` was used -- save each condition's CSV
under a distinct name (e.g. `phase2_deltaofv_free.csv` /
`phase2_deltaofv_amortized.csv`) before running the next condition.

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

## `publication/` — manuscript tables and figures

Both scripts follow the same design: read whichever result CSVs you have,
skip gracefully (with a clear `[skip]` message) whenever a source isn't
supplied or doesn't exist, and never guess which condition a file
corresponds to.

**Why explicit file paths for everything, not default filenames**: several
scripts above write to a fixed default filename regardless of which
condition was run (`nlme_vi_phase2_realdata.py` always writes
`phase2_realdata_results.csv` whether you ran Theoph or warfarin;
`nlme_vi_phase2_deltaofv.py` always writes `phase2_deltaofv_results.csv`
whether `--posterior` was `free` or `amortized`). If you ran multiple
conditions, you need to have already saved/renamed each one immediately
after that run (see the CSV-overwrite gotcha called out in each section
above) -- these scripts take one explicit `--<condition>-csv` flag per
named result set rather than assuming a fixed layout, specifically so a
missing save doesn't silently produce a table from the wrong condition.

```bash
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

python publication/make_figures.py \
    --phase0-csv outputs/phase0_results.csv \
    --phase1-csv outputs/phase1_results.csv \
    --nonlinear-csv outputs/phase1_nonlinear_results.csv \
    --theoph-csv outputs/phase2_realdata_theoph.csv \
    --warfarin-csv outputs/phase2_realdata_warfarin.csv \
    --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
    --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized.csv \
    --psis-csv outputs/phase2_psis_results.csv \
    --out publication/figures
```

Every flag is optional -- run with whatever you have; tables/figures fill
in as more results become available, nothing errors on a missing source.

**`make_tables.py`** produces, for each available source: Phase 0 headline
(Omega bias by posterior x K), Phase 1 Q2/Q4 grid (linear scenarios),
Phase 1 Q3 (nonlinear tier, all parameters not just omegas), real-data
K=1-vs-K=high (one row per dataset per parameter), dOFV calibration
(boundary fraction, KS stat/p, type-I error under the correct reference),
PSIS/ESS aggregate summary, and VI-vs-FOCEI/SAEM (only if the baseline
comparison exists). Each table is written as both `.csv` (further
processing) and `.md` (direct copy-paste into a manuscript draft), no
extra dependencies (markdown writing is done manually, not via
`DataFrame.to_markdown()`, which requires the separate `tabulate`
package).

**`make_figures.py`** produces the matching figures. **Deliberately no
captions or descriptive titles baked into the images** (no
`fig.suptitle()` with a "what this shows" sentence) -- only the axis
labels, legends, and panel labels needed for a plot to be interpretable on
its own; caption text is left for the manuscript itself. 300 DPI PNG
output.

**`reproduce.sh` does not exist yet** -- planned as a single script that
re-runs everything end to end from a clean environment, for reviewers.
Deliberately last on the list: building it before the numbers above are
final would mean rebuilding it every time a result changes.

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
  above. Not blocking anything else. Also doesn't implement the nonlinear
  (MM) model -- it's hardcoded to the linear 1-compartment structure.
  Deliberately not extended: not needed for this paper's claims (Q3 is
  answered by the VI-internal K=1-vs-K=64 comparison, no external baseline
  required), and would be a real, separate coding task (new `model()`
  block, likely its own debugging cycle) if ever wanted for other purposes.
- **Amortized+flow (Phase 1) is unstable at every tested K**, not just
  K=1 -- see Key Finding 3. Not load-bearing for any core claim, excluded
  from headline results at every K. Free+flow and free/amortized+gaussian
  are unaffected and stable.
- **dOFV residual calibration gap**: real, confirmed at n=100 and n=300,
  source not identified. The boundary-mass component is fixed
  (amortized-gaussian); a smaller residual deviation in the non-boundary
  shape persists. A sigma-divergence hypothesis was tested and found
  correlated but not conclusively causal (mechanically coupled with the
  likelihood by construction -- see Key Finding 5 and the deltaofv
  section above). Genuinely open; reportable as a stated limitation as-is.
- **dOFV / amortized posterior non-convergence**: 23-27% non-convergence
  rate, confirmed NOT the primary driver of the residual gap above, but a
  separate quality issue worth its own fix (likely needs higher
  `--max-steps` for this specific combination -- not yet tried).
- **Nonlinear tier production run**: COMPLETE. free+gaussian is the clean,
  reportable result (see Key Finding 6). amortized+gaussian failed
  catastrophically and is excluded from headline results -- a new
  instability, not the already-known amortized+flow one. Not investigated
  further; free+gaussian alone answers Q3. `commands.sh`'s
  `phase1_nonlinear_production` restricts to `--posteriors free` for this
  reason (previously ran both).
- **Runtime, for future planning**: 120 fits, 9-way parallel, ~88 CPU-hours
  total, ~10-11 hours real elapsed time -- consistent with the cost model's
  prediction (5-24h). The failed amortized cells cost roughly as much as
  the successful free cells (non-convergence still runs the full
  `--max-steps` budget), so excluding amortized going forward
  meaningfully cuts future nonlinear-tier cost too, not just risk.
- **CSV-overwrite gotcha (cross-cutting, applies to `nlme_vi_phase2_realdata.py`
  and `nlme_vi_phase2_deltaofv.py`)**: both scripts always write to the same
  default filename regardless of which condition (`--dataset`, `--posterior`)
  was run. Running a second condition without saving the first's output
  first **silently overwrites it** -- this already happened once during
  development (a warfarin run overwrote an unsaved Theoph result). Copy
  each condition's CSV to a distinct name immediately after that run
  completes, before starting the next one. `publication/make_tables.py`
  and `make_figures.py` are built around this -- they take one explicit
  path per named condition rather than assuming a fixed layout.
- **MPS (Apple Silicon GPU) does not support float64** -- a permanent
  Metal backend limitation, confirmed by hitting it directly, not assumed
  in advance. This project requires float64 throughout. `set_device()`
  raises immediately and clearly for `mps` (or `auto` detecting it) rather
  than failing deep in model construction. CPU is the only supported
  device on Apple Silicon for this codebase; there is no float64
  workaround. `cuda` remains untested for the RK4 nonlinear tier
  specifically (no CUDA GPU available during development).
- **N-scaling small-sample-bias finding**: currently 1 replicate per N
  (12/40/80, and separately 15/40/80 for the nonlinear tier's Vmax/V).
  Directionally clear and consistent across both the linear and nonlinear
  tiers (bias shrinks as N grows, holding K fixed) but needs 5-10
  replicates per N before citing the magnitude precisely.
- **`reproduce.sh` doesn't exist yet** -- `make_tables.py`/`make_figures.py`
  are done (see the `publication/` section above). `reproduce.sh` is
  deliberately last: a single reviewer-facing script to run everything end
  to end, which only makes sense to write once the numbers it reproduces
  are actually final.
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
- **`--n-workers` (`nlme_vi_phase2_deltaofv.py` and `nlme_vi_phase1.py`)**:
  fits/replicates are independent: `os.cpu_count() - 1` is a reasonable
  default. Each worker is pinned to 1 PyTorch thread internally to avoid
  CPU oversubscription -- don't remove this if extending the pattern
  elsewhere. Both scripts require `if __name__ == "__main__":` for this to
  work correctly (see each script's section above) -- a real, if minor,
  deviation from the pure cell-based style used elsewhere in this repo.
- **`--device`**: leave at the default `cpu`. MPS (Apple Silicon) doesn't
  support float64 and will raise immediately rather than fail silently or
  deep in model construction -- this isn't a bug to work around, there's
  no fix available on the MPS side.
- **Save condition-specific CSVs immediately, not "later."** Several
  scripts share one default output filename across different conditions
  (dataset, posterior) -- the second run silently overwrites the first.
  This isn't hypothetical: it happened once during development. Copy the
  file to a distinct name the moment a run finishes, before starting the
  next one.
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
