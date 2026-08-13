#!/usr/bin/env bash
# reproduce.sh -- regenerates every table and figure in the manuscript,
# end to end, from a clean checkout.
#
# UNLIKE commands.sh, this IS a straight-line pipeline, deliberately. Every
# flag below is locked to the exact settings the reported numbers use --
# this script is for a reviewer (or future-you) who wants mechanical
# reproduction, not exploration. If you want to explore/vary settings, use
# commands.sh or call the underlying scripts directly instead.
#
# NOT resumable/checkpointed. If interrupted, either restart from scratch
# or comment out the sections already completed and rerun -- there is no
# automatic skip-what's-done logic.
#
# Expected total runtime: several hours to about a day, DOMINATED by the
# nonlinear grid (5-24h, see README's cost estimate) and the deltaofv
# n=300 confirmatory run (roughly 1-2h with parallelism). Everything else
# combined is well under an hour. Run with `caffeinate -i` on macOS or
# expect to babysit it.
#
# Usage: bash reproduce.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root -- this script lives in publication/

OUT=outputs
mkdir -p "$OUT" "$OUT/phase2_baselines" publication/tables publication/figures

n_cores() {
    if command -v nproc >/dev/null 2>&1; then nproc
    elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu
    else echo 4
    fi
}
WORKERS=$(( $(n_cores) - 1 ))
[[ "$WORKERS" -lt 1 ]] && WORKERS=1

CAFFEINATE=""
command -v caffeinate >/dev/null 2>&1 && CAFFEINATE="caffeinate -i"

echo "================================================================"
echo "REPRODUCE.SH -- full manuscript pipeline"
echo "Workers for parallelized steps: $WORKERS"
echo "Expected runtime: several hours, dominated by the nonlinear grid"
echo "and the deltaofv n=300 run. NOT resumable -- see header comment."
echo "================================================================"

# ============================================================ Phase 0
echo -e "\n>>> Phase 0"
python nlme_vi_phase0.py --reps 8 --subjects 120 --max-steps 25000 --out "$OUT"

# ============================================================ Phase 1: linear tier
echo -e "\n>>> Phase 1, Q1 (convergence check)"
python nlme_vi_phase1.py --converge-check --converge-steps 25000 --out "$OUT"

echo -e "\n>>> Phase 1, Q2/Q4 gaussian (both posteriors stable)"
python nlme_vi_phase1.py --scenarios dense,sparse --families gaussian \
    --posteriors free,amortized --K 1,8,64 --reps 30 --out "$OUT"
cp "$OUT/phase1_results.csv" "$OUT/phase1_linear_gaussian.csv"

echo -e "\n>>> Phase 1, Q4 flow (free posterior only -- amortized+flow is unstable"
echo "    at every K, see README Key Finding 3; NOT run here on purpose)"
python nlme_vi_phase1.py --scenarios dense,sparse --families flow \
    --posteriors free --K 1,8,64 --reps 30 --out "$OUT"
cp "$OUT/phase1_results.csv" "$OUT/phase1_linear_flow.csv"

# Combine gaussian + flow into one file for the tables/figures scripts,
# which expect a single --phase1-csv covering the full Q2/Q4 grid.
python3 -c "
import pandas as pd
g = pd.read_csv('$OUT/phase1_linear_gaussian.csv')
f = pd.read_csv('$OUT/phase1_linear_flow.csv')
pd.concat([g, f], ignore_index=True).to_csv('$OUT/phase1_linear_combined.csv', index=False)
"

# ============================================================ Phase 1: nonlinear tier
echo -e "\n>>> Phase 1, Q3 nonlinear -- gaussian only, free+amortized. This is"
echo "    the expensive step (see header). Flow deliberately excluded: never"
echo "    validated on this tier, and not needed -- Vmax/V's shrinkage-and-"
echo "    correction signature is already established via gaussian alone."
$CAFFEINATE python nlme_vi_phase1.py --scenarios nonlinear --families gaussian \
    --posteriors free,amortized --nl-reps 20 --nl-subjects 60 --mm-dt 0.1 \
    --n-workers "$WORKERS" --out "$OUT"
cp "$OUT/phase1_results.csv" "$OUT/phase1_nonlinear_results.csv"

# ============================================================ Phase 2: real data
echo -e "\n>>> Phase 2, real data: Theophylline"
uv run phase2/nlme_vi_phase2_realdata.py --dataset theoph --K 1,64 \
    --max-steps 40000 --out "$OUT"
cp "$OUT/phase2_realdata_results.csv" "$OUT/phase2_realdata_theoph.csv"

echo -e "\n>>> Phase 2, real data: warfarin"
if [[ ! -f warfarin.csv ]]; then
    echo "ERROR: warfarin.csv not found. This one step cannot be automated --"
    echo "export it from R first (no compilation needed, pure data package):"
    echo '  R> install.packages("nlmixr2data")'
    echo '  R> write.csv(nlmixr2data::warfarin, "warfarin.csv", row.names = FALSE)'
    echo "Then rerun this script (earlier steps will simply repeat -- see"
    echo "header re: no resumability -- or comment out completed sections)."
    exit 1
fi
python3 -c "
import pandas as pd
df = pd.read_csv('warfarin.csv')
pk = df[df.dvid == 'cp']
pk.to_csv('warfarin_pk.csv', index=False)
print(f'{len(pk)} PK rows extracted (expect 283 for the standard dataset)')
"
uv run phase2/nlme_vi_phase2_realdata.py --csv warfarin_pk.csv \
    --col-map "id=subject,time=time,dv=conc,amt=dose" --K 1,64 --max-steps 40000 --out "$OUT"
cp "$OUT/phase2_realdata_results.csv" "$OUT/phase2_realdata_warfarin.csv"

# ============================================================ Phase 2: PSIS
echo -e "\n>>> Phase 2, PSIS/ESS diagnostic"
python phase2/nlme_vi_phase2_psis.py --subjects 120 --bad-steps 30 --out "$OUT"

# ============================================================ Phase 2: dOFV
echo -e "\n>>> Phase 2, dOFV -- free posterior (n=100, reproduces the miscalibration)"
$CAFFEINATE uv run phase2/nlme_vi_phase2_deltaofv.py \
    --reps 100 --subjects 120 --K 64 --n-workers "$WORKERS" --out "$OUT"
cp "$OUT/phase2_deltaofv_results.csv" "$OUT/phase2_deltaofv_free.csv"

echo -e "\n>>> Phase 2, dOFV -- amortized posterior (n=300, confirmatory)"
$CAFFEINATE uv run phase2/nlme_vi_phase2_deltaofv.py \
    --reps 300 --subjects 120 --K 64 --posterior amortized \
    --n-workers "$WORKERS" --out "$OUT"
cp "$OUT/phase2_deltaofv_results.csv" "$OUT/phase2_deltaofv_amortized_n300.csv"

# ============================================================ Phase 2: FOCEI/SAEM baseline
echo -e "\n>>> Phase 2, VI vs FOCEI/SAEM baseline (optional -- skips gracefully"
echo "    if R/nlmixr2 isn't set up; see README Requirements)"
if command -v Rscript >/dev/null 2>&1 && Rscript -e 'library(nlmixr2)' >/dev/null 2>&1; then
    uv run phase2/nlme_vi_phase2_baselines.py --subjects 120 --reps 5 \
        --out "$OUT/phase2_baselines"
    BASELINE_CSV="$OUT/phase2_baselines/phase2_baseline_comparison.csv"
else
    echo "R/nlmixr2 not available -- skipping. This does not affect any other"
    echo "result (see README: no core claim depends on this baseline)."
    BASELINE_CSV=""
fi

# ============================================================ Tables + figures
echo -e "\n>>> Generating manuscript tables"
python publication/make_tables.py \
    --phase0-csv "$OUT/phase0_results.csv" \
    --phase1-csv "$OUT/phase1_linear_combined.csv" \
    --nonlinear-csv "$OUT/phase1_nonlinear_results.csv" \
    --theoph-csv "$OUT/phase2_realdata_theoph.csv" \
    --warfarin-csv "$OUT/phase2_realdata_warfarin.csv" \
    --deltaofv-free-csv "$OUT/phase2_deltaofv_free.csv" \
    --deltaofv-amortized-csv "$OUT/phase2_deltaofv_amortized_n300.csv" \
    --psis-csv "$OUT/phase2_psis_results.csv" \
    --baseline-csv "$BASELINE_CSV" \
    --out publication/tables

echo -e "\n>>> Generating manuscript figures"
python publication/make_figures.py \
    --phase0-csv "$OUT/phase0_results.csv" \
    --phase1-csv "$OUT/phase1_linear_combined.csv" \
    --nonlinear-csv "$OUT/phase1_nonlinear_results.csv" \
    --theoph-csv "$OUT/phase2_realdata_theoph.csv" \
    --warfarin-csv "$OUT/phase2_realdata_warfarin.csv" \
    --deltaofv-free-csv "$OUT/phase2_deltaofv_free.csv" \
    --deltaofv-amortized-csv "$OUT/phase2_deltaofv_amortized_n300.csv" \
    --psis-csv "$OUT/phase2_psis_results.csv" \
    --out publication/figures

echo -e "\n================================================================"
echo "DONE. Tables -> publication/tables/, figures -> publication/figures/"
echo "Raw results (each condition saved under a distinct name) -> $OUT/"
echo "================================================================"