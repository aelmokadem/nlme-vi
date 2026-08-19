# NLME Variational Inference

Characterizing and correcting underestimation of **between-subject variability (BSV)** by variational inference (VI) in nonlinear mixed-effects (NLME) models, and evaluating when importance-sampling-based marginal likelihoods support downstream likelihood-based inference.

The central result is simple: the standard evidence lower bound (**ELBO; $K=1$**) can substantially underestimate random-effect standard deviations, whereas a tighter importance-weighted objective (**IW-ELBO; $K>1$**) markedly reduces this bias. The repository tests that result across sampling designs, posterior architectures, variational families, real pharmacokinetic data, a nonlinear Michaelis-Menten model, and matched FOCEI/SAEM benchmarks.

> **Terminology.** Throughout this repository, $\Omega$ denotes the random-effect covariance matrix and $\omega$ denotes a random-effect **standard deviation**. In prose and tables, **BSV** means between-subject variability and **RUV** means residual unexplained variability.

---

## Key findings

1. **Standard-ELBO VI underestimates BSV.** On simulated data with known ground truth, $K=1$ systematically underestimates random-effect SDs. Increasing $K$ to 8 or 64 substantially reduces the bias.

2. **Sparse sampling worsens the effect.** At matched $K$, sparse designs produce greater BSV shrinkage than dense designs, consistent with weaker subject-level identifiability amplifying the variational approximation error.

3. **A richer posterior family can reduce low\-$K$ bias.** Free-posterior normalizing flows substantially reduce BSV bias in well-identified designs, including at $K=1$.

4. **Amortized + flow is unstable in the architecture tested here.** The encoder output was used both to parameterize the flow base distribution and to condition the flow transformation. This configuration was unstable across $K=1,8,64$ and is excluded from primary results. A decoupled fixed-base design is a promising future direction but has not been evaluated at production scale.

5. **The pattern reproduces on real data.** Theophylline and warfarin both show smaller random-effect SD estimates at $K=1$ than at $K=64$ on the same dataset. Warfarin fixed-effect estimates for clearance and volume are also compatible with published nlmixr2 FOCEI results despite differences in absorption-model structure.

6. **The correction survives a genuinely nonlinear model.** In a one-compartment IV Michaelis-Menten model with BSV on $V_{\max}$ and $V$, free-Gaussian VI reproduces the $K=1$-to-$K=64$ correction. Preliminary experiments assigning BSV simultaneously to $V_{\max}$ and $K_m$ showed severe instability in the $K_m$ random effect, consistent with an identifiability limitation rather than a VI-specific effect.

7. **Amortized-Gaussian VI has a separate nonlinear failure mode.** On the Michaelis-Menten tier, the amortized Gaussian posterior became less stable as $K$ increased. Because no flow is involved, this is distinct from the amortized-flow failure. Candidate explanations include degradation of inference-network gradient signal-to-noise under importance-weighted training and a harder amortized representation problem across qualitatively different nonlinear trajectories. These explanations remain hypotheses rather than established mechanisms.

8. **Importance-sampling ESS is a useful fit diagnostic.** Aggregate importance-sampling effective sample size (ESS) clearly separates well-converged from deliberately under-trained fits on matched simulated data. Aggregate ESS is more useful than a fixed per-subject tail threshold.

9. **Recovered likelihoods are not automatically valid for every LRT.** When nested models differ in random-effect structure, likelihood-ratio tests based on a free posterior are miscalibrated relative to the Self-Liang boundary-mixture reference. A shared amortized posterior substantially restores the expected boundary mass, although a smaller residual deviation in the positive component remains unresolved.

10. **Corrected VI reaches FOCEI/SAEM-comparable accuracy on the matched linear benchmark.** Across 20 simulated datasets with $N=120$, VI at $K=64$ recovers every reported random-effect SD with bias below 1.2%, comparable to FOCEI and SAEM. VI at $K=1$ retains the characteristic BSV underestimation.

11. **The runtime result is deliberately narrow.** On the simple analytic benchmark, mean process CPU time was approximately 11.25 s for FOCEI, 12.68 s for SAEM, 5.14 s for VI at $K=1$, and 16.50 s for VI at $K=64$. Thus, correcting the VI bias is not computationally free: $K=64$ VI was slower than FOCEI/SAEM on this simple model. Thread counts were not explicitly matched between PyTorch and R/BLAS, so this is not intended as a general speed benchmark.

---

## Repository layout

```text
.
├── nlmevi_core.py
├── nlme_vi_phase0.py
├── nlme_vi_phase1.py
├── commands.sh
├── manuscript_copyedited.qmd
├── references.bib
│
├── phase2/
│   ├── nlme_vi_phase2_deltaofv.py
│   ├── nlme_vi_phase2_psis.py
│   ├── baseline_nlmixr2.R
│   ├── nlme_vi_phase2_baselines.py
│   └── nlme_vi_phase2_realdata.py
│
├── publication/
│   ├── make_tables.R
│   ├── make_figures.R
│   ├── tables/
│   └── figures/
│
└── outputs/
```

`nlmevi_core.py` contains the shared models, posterior implementations, IW-ELBO objective, importance-sampling likelihood evaluation, device handling, and adaptive training machinery.

If the Phase 2 scripts import the core module from inside `phase2/`, create the symlink once:

```bash
ln -s ../nlmevi_core.py phase2/nlmevi_core.py
```

---

## Requirements

### Python

```bash
pip install torch numpy pandas scipy matplotlib rdatasets
```

### R

R is used for FOCEI/SAEM benchmarking through `nlmixr2` and for the manuscript-facing table and figure pipeline.

```r
install.packages(c(
  "nlmixr2", "nlmixr2data",
  "dplyr", "readr", "tidyr", "purrr", "stringr",
  "ggplot2", "patchwork", "here"
))
```

The completed benchmark reported in the manuscript used R 4.5.3, nlmixr2 7.0.1, nlmixr2est 7.0.2, and rxode2 5.1.6 on macOS (aarch64-apple-darwin20).

---

## Core implementation: `nlmevi_core.py`

| Component | Purpose |
|---|---|
| `OneCmtOral` | Linear one-compartment oral PK model with analytic solution |
| `OneCmtIVBolusMM` | Original Michaelis-Menten model implementation |
| `OneCmtIVBolusMMNoKmRE` | Primary nonlinear model; BSV on $V_{\max}$ and $V$, not $K_m$ |
| `FreePosterior` | Per-subject Gaussian variational parameters |
| `AmortizedPosterior` | Shared encoder producing subject-level variational parameters |
| `FlowPosterior` | Conditional normalizing-flow posterior |
| `AffineCoupling` / `ConditionalFlow` | Flow transformation machinery |
| `iw_elbo` | $K$-sample importance-weighted ELBO |
| `is_marginal_loglik` | Post-hoc importance-sampling marginal likelihood and weight diagnostics |
| `train_to_convergence` | Adaptive convergence based on stabilization of random-effect SD estimates |
| `fit_model` | Common fitting interface used by downstream experiments |
| `set_device` | CPU/CUDA/MPS device handling |

### Training $K$ vs evaluation $K$

The **training $K$** controls the IW-ELBO used to estimate model and variational parameters. The **evaluation $K$** (`eval_k`) controls the number of importance samples used after training to estimate the marginal likelihood more precisely. Increasing `eval_k` does **not** retrain the model.

### Device support

The project uses float64 throughout because high-$K$ importance weights can span many orders of magnitude. Apple's MPS backend does not support the required float64 operations, so Apple Silicon runs should use CPU. CUDA remains the appropriate GPU target where compatible hardware is available.

---

## Experimental design rule

> **One seed generates one dataset, and every method arm in a comparison is fit to that byte-identical dataset.**

Do not regenerate data independently for different $K$, posterior, family, or estimation-method arms.

---

## Phase 0: establish the BSV-bias signal

`nlme_vi_phase0.py` asks whether standard-ELBO VI underestimates BSV and whether increasing $K$ corrects it.

```bash
python nlme_vi_phase0.py --quick
python nlme_vi_phase0.py
```

The full analysis evaluates free and amortized Gaussian posteriors across $K=1,8,64$ on the linear one-compartment oral model.

Primary result: BSV is strongly $K$-dependent, whereas structural fixed effects are much less sensitive. In particular, $k_a$ remains somewhat underestimated across $K$, showing that increasing $K$ is not a generic correction for every form of parameter bias.

---

## Phase 1: stress tests

`nlme_vi_phase1.py` evaluates dense vs sparse sampling, Gaussian vs flow families, free vs amortized posterior architectures, and nonlinear Michaelis-Menten elimination.

```bash
python nlme_vi_phase1.py   --scenarios dense,sparse   --families gaussian   --posteriors free,amortized   --K 1,8,64   --reps 30
```

For flow analyses, the primary reportable configuration is the **free posterior**:

```bash
python nlme_vi_phase1.py   --scenarios dense,sparse   --families flow   --posteriors free   --K 1,8,64   --reps 30
```

The amortized-flow implementation tested here is excluded from primary results because of instability across all tested $K$.

### Nonlinear tier

The primary nonlinear analysis uses `OneCmtIVBolusMMNoKmRE`, with BSV on $V_{\max}$ and $V$, while $K_m$ is estimated as a fixed effect.

```bash
caffeinate -i python nlme_vi_phase1.py   --scenarios nonlinear   --families gaussian   --posteriors free   --nl-reps 20   --nl-subjects 60   --mm-dt 0.1   --n-workers 9   --out .
```

Amortized-Gaussian fits were also explored on this tier and showed substantial instability that worsened with increasing $K$. Two candidate mechanisms are discussed in the manuscript: inference-network gradient degradation under importance-weighted training and a harder shared-encoder representation problem. Neither has been established causally.

---

## Phase 2: real-data validation

`phase2/nlme_vi_phase2_realdata.py` tests whether the simulation-derived $K=1$-vs-$K=64$ BSV pattern appears on real PK data.

### Theophylline

```bash
uv run phase2/nlme_vi_phase2_realdata.py   --dataset theoph   --K 1,64   --max-steps 40000
```

### Warfarin

Export `nlmixr2data::warfarin` from R if needed, then run the Python analysis with the appropriate column mapping.

Both datasets reproduce the expected direction: $K=1$ gives smaller random-effect SD estimates than $K=64$ for at least some BSV parameters.

Three data-handling lessons from this work should be preserved when extending the loader:

- remove pre-dose observations based on **time**, not merely non-positive concentration;
- derive initialization from the scale of the dataset;
- extract dose from the original subject data in a way that supports both repeated-dose-covariate and NONMEM-style `AMT` conventions.

**Output warning:** the real-data script uses a common default filename across datasets. Save each result immediately under a dataset-specific filename before running the next condition.

---

## Phase 2: importance-sampling ESS

```bash
python phase2/nlme_vi_phase2_psis.py   --subjects 120   --bad-steps 30
```

The principal diagnostic is **aggregate importance-sampling ESS**. Mean or median ESS separates well-converged and deliberately under-trained fits more cleanly than a fixed per-subject tail-count threshold.

---

## Phase 2: likelihood-ratio calibration

`phase2/nlme_vi_phase2_deltaofv.py` evaluates random-effect selection under the Self-Liang boundary-mixture reference,

$$
\frac{1}{2}\delta_0 + \frac{1}{2}\chi^2_1.
$$

Free posterior:

```bash
caffeinate -i uv run phase2/nlme_vi_phase2_deltaofv.py   --reps 100 --subjects 120 --K 64 --n-workers 9
```

Amortized posterior:

```bash
caffeinate -i uv run phase2/nlme_vi_phase2_deltaofv.py   --reps 100 --subjects 120 --K 64 --n-workers 9   --posterior amortized
```

The free posterior is miscalibrated for comparisons that change random-effect structure. The amortized posterior substantially restores the expected ~50% boundary mass, although a residual discrepancy remains in the positive component of the null distribution.

This finding does **not** invalidate the BSV-estimation analyses, which do not depend on this likelihood-ratio test.

---

## FOCEI / SAEM benchmark

The matched baseline comparison is implemented through:

- `phase2/baseline_nlmixr2.R`
- `phase2/nlme_vi_phase2_baselines.py`

The final manuscript benchmark uses $N=120$ subjects and 20 replicate datasets. Each replicate is fit by FOCEI, SAEM, VI at $K=1$, and VI at $K=64$ on identical data.

Direct R execution:

```bash
Rscript phase2/baseline_nlmixr2.R   outputs/phase2_baselines/rep0_data.csv   outputs/phase2_baselines/rep0_foce.csv   foce

Rscript phase2/baseline_nlmixr2.R   outputs/phase2_baselines/rep0_data.csv   outputs/phase2_baselines/rep0_saem.csv   saem
```

### Accuracy

FOCEI and SAEM recover the BSV parameters with small bias. VI at $K=64$ achieves similarly small bias, below 1.2% for every reported random-effect SD. VI at $K=1$ substantially underestimates $\omega_V$ and $\omega_{k_a}$.

### Runtime

Approximate manuscript means using process CPU time:

| Method | Mean CPU time |
|---|---:|
| FOCEI | 11.25 s |
| SAEM | 12.68 s |
| VI, $K=1$ | 5.14 s |
| VI, $K=64$ | 16.50 s |

These values apply only to the simple analytic benchmark. Thread counts were not explicitly pinned between PyTorch and R/BLAS.

---

## Publication workflow

The manuscript-facing publication pipeline is implemented in **R/tidyverse**.

### Tables

```bash
bash commands.sh publication_tables_r
```

Outputs are written under:

```text
publication/tables/
```

Table terminology follows the manuscript:

- **BSV** = between-subject variability
- **RUV** = residual unexplained variability
- BSV parameters $\omega$ are random-effect standard deviations

### Figures

```bash
bash commands.sh publication_figures_r
```

Outputs are written under:

```text
publication/figures/
```

Scientific captions live in the Quarto manuscript rather than being baked into the images.

The current manuscript uses primary figures for Phase 0 BSV bias, the dense/sparse Phase 1 grid, nonlinear BSV bias, real-data validation, ESS diagnostics, and $\Delta$OFV calibration. Supplementary figures show companion fixed-effect/RUV results.

### Python publication scripts

Earlier Python versions may remain useful as tested reference implementations, but the **R/tidyverse scripts are the manuscript-facing publication pipeline**. Avoid maintaining two independently evolving definitions of final manuscript outputs.

---

## Manuscript

The current Quarto manuscript is:

```text
manuscript_copyedited.qmd
```

Render all configured formats with:

```bash
quarto render manuscript_copyedited.qmd
```

The manuscript is the source of truth for formal scientific interpretation, citations, limitations, and the distinction between primary and exploratory results. The README is intentionally more implementation-focused.

---

## `commands.sh`

`commands.sh` provides named shell functions for common analyses.

```bash
bash commands.sh
bash commands.sh phase1_gaussian_grid
bash commands.sh publication_tables_r
bash commands.sh publication_figures_r
```

For long macOS runs, use `caffeinate -i` where appropriate.

---

## Known limitations and open questions

- **Amortized-flow architecture:** unstable across $K$ in the architecture tested here. A fixed-standard-normal-base design is a promising future direction.
- **Nonlinear amortized-Gaussian instability:** deterioration with increasing $K$ is real, but the proposed gradient-SNR and representation-capacity explanations remain unconfirmed.
- **Michaelis-Menten identifiability:** simultaneous BSV on $V_{\max}$ and $K_m$ was unstable; the primary nonlinear analysis uses BSV on $V_{\max}$ and $V$ only.
- **LRT residual calibration:** amortization substantially corrects the boundary-mass problem, but the positive component remains imperfectly calibrated.
- **Runtime scope:** the benchmark covers only a simple analytic linear model, and PyTorch/R-BLAS thread counts were not explicitly matched.
- **Model scope:** generalization to multicomponent models, correlated random effects, covariate models, time-varying dosing, and additional RUV structures remains to be established.
- **Nonlinear cross-method benchmark:** direct FOCEI/SAEM comparison on the Michaelis-Menten model remains future work.
- **Reproduction wrapper:** a single reviewer-facing `reproduce.sh` would be useful once all final manuscript inputs and expensive analyses are frozen.

---

## Practical notes

### Use CPU time for runtime comparisons

Use process CPU-time fields rather than wall-clock time for scientific runtime comparisons. Wall time can be severely distorted by system sleep or unrelated process contention.

### Prevent macOS sleep

```bash
caffeinate -i <command>
```

### Preserve condition-specific outputs

Several scripts use the same default filename for multiple conditions. Rename or copy outputs immediately after each run.

### Parallel execution

Independent simulation replicates can be parallelized with `--n-workers`. On macOS, scripts using `ProcessPoolExecutor` require the standard:

```python
if __name__ == "__main__":
```

guard because workers are spawned by re-importing the module.

### Do not silently change the statistical model

When extending FOCEI/SAEM or VI code, ensure that the structural model, parameterization, BSV structure, RUV model, input data, and initialization philosophy remain matched across methods.

---

## Scientific interpretation in one paragraph

This repository shows that standard-ELBO VI can systematically shrink estimated between-subject variability in NLME models even when structural fixed effects are reasonably recovered. Tightening the variational objective through importance weighting substantially corrects that effect, and $K=64$ VI achieves FOCEI/SAEM-comparable parameter accuracy in the matched linear benchmark. Richer posterior families can also reduce the bias, but posterior architecture matters: some amortized configurations introduce distinct optimization or representation failures, and accurate point estimation does not by itself guarantee calibrated likelihood-ratio inference. Importance-sampling ESS provides a practical diagnostic of posterior fit quality, while the nonlinear and real-data experiments show that the central BSV result extends beyond the simplest simulated setting.

---

## Citation

This repository accompanies the manuscript:

> *Importance-Weighted Variational Inference for Nonlinear Mixed-Effects Models: Characterizing and Correcting Bias in Between-Subject Variability.*

Citation details can be added after publication or preprint deposition.
