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

set -uo pipefail   # NOTE: -e deliberately NOT set here -- see run_step() below.
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

# Fault tolerance: without this, ANY single failing step (a transient
# environment issue on a reviewer's machine, a real bug hit on data this
# specific run happens to generate, etc.) would halt the ENTIRE script via
# plain `set -e` -- including the final table/figure generation, even for
# results that already succeeded and don't depend on whatever failed. This
# defeats the whole point of make_tables.py/make_figures.py being designed
# to gracefully skip missing inputs. run_step() catches a failure, logs it
# clearly, records it, and lets the script continue -- the final summary
# reports what failed, and table/figure generation always runs at the end
# with whatever files actually exist.
FAILED_STEPS=()
run_step() {
    local step_name="$1"; shift
    echo -e "\n>>> $step_name"
    if "$@"; then
        return 0
    else
        echo "    *** FAILED: $step_name -- continuing with remaining steps ***"
        FAILED_STEPS+=("$step_name")
        return 1
    fi
}

echo "================================================================"
echo "REPRODUCE.SH -- full manuscript pipeline"
echo "Workers for parallelized steps: $WORKERS"
echo "Expected runtime: several hours, dominated by the nonlinear grid"
echo "and the deltaofv n=300 run. NOT resumable -- see header comment."
echo "A failed step does NOT halt the script -- see run_step() above."
echo "================================================================"

# Checked FIRST, before any expensive computation, specifically so a
# reviewer without R/nlmixr2data set up finds out in SECONDS, not after
# 5-24+ hours of Phase 0/1 compute. This does NOT halt the script (unlike
# an earlier version of this file) -- warfarin is simply skipped later,
# same graceful-degradation pattern already used for the R FOCEI/SAEM
# baseline below. Every other result in the paper is independent of this.
HAVE_WARFARIN=0
if [[ -f warfarin.csv ]]; then
    HAVE_WARFARIN=1
else
    echo ""
    echo ">>> NOTE: warfarin.csv not found -- the warfarin real-data result"
    echo "    will be skipped (Theophylline and everything else is unaffected)."
    echo "    To include it, export from R first (no compilation needed,"
    echo "    pure data package), then rerun this script:"
    echo '      R> install.packages("nlmixr2data")'
    echo '      R> write.csv(nlmixr2data::warfarin, "warfarin.csv", row.names = FALSE)'
fi

# ============================================================ Phase 0
run_step "Phase 0" python nlme_vi_phase0.py --reps 8 --subjects 120 --max-steps 25000 --out "$OUT" || true

# ============================================================ Phase 1: linear tier
run_step "Phase 1, Q1 (convergence check)" \
    python nlme_vi_phase1.py --converge-check --converge-steps 25000 --out "$OUT" || true

if run_step "Phase 1, Q2/Q4 gaussian (both posteriors stable)" \
    python nlme_vi_phase1.py --scenarios dense,sparse --families gaussian \
        --posteriors free,amortized --K 1,8,64 --reps 30 --out "$OUT"; then
    cp "$OUT/phase1_results.csv" "$OUT/phase1_linear_gaussian.csv"
fi

echo "    (Q4 flow: free posterior only -- amortized+flow is unstable at every"
echo "    K, see README Key Finding 3; NOT run here on purpose)"
if run_step "Phase 1, Q4 flow" \
    python nlme_vi_phase1.py --scenarios dense,sparse --families flow \
        --posteriors free --K 1,8,64 --reps 30 --out "$OUT"; then
    cp "$OUT/phase1_results.csv" "$OUT/phase1_linear_flow.csv"
fi

# Combine gaussian + flow into one file for the tables/figures scripts,
# which expect a single --phase1-csv covering the full Q2/Q4 grid. Only
# attempted if BOTH pieces actually exist -- if either step above failed,
# this is silently skipped rather than combining a partial/wrong result;
# make_tables.py/make_figures.py will just skip the linear-grid outputs
# downstream, same as any other missing file.
if [[ -f "$OUT/phase1_linear_gaussian.csv" && -f "$OUT/phase1_linear_flow.csv" ]]; then
    python3 -c "
import pandas as pd
g = pd.read_csv('$OUT/phase1_linear_gaussian.csv')
f = pd.read_csv('$OUT/phase1_linear_flow.csv')
pd.concat([g, f], ignore_index=True).to_csv('$OUT/phase1_linear_combined.csv', index=False)
"
fi

# ============================================================ Phase 1: nonlinear tier
echo "    (Q3 nonlinear: gaussian/free ONLY. Flow never validated on this"
echo "    tier; amortized+gaussian tested at full scale and FAILED"
echo "    catastrophically -- bias +230% to +1168%, worsening with K, a new"
echo "    instability distinct from amortized+flow. See README Key Finding 6.)"
if run_step "Phase 1, Q3 nonlinear" \
    $CAFFEINATE python nlme_vi_phase1.py --scenarios nonlinear --families gaussian \
        --posteriors free --nl-reps 20 --nl-subjects 60 --mm-dt 0.1 \
        --n-workers "$WORKERS" --out "$OUT"; then
    cp "$OUT/phase1_results.csv" "$OUT/phase1_nonlinear_results.csv"
fi

# ============================================================ Phase 2: real data
if run_step "Phase 2, real data: Theophylline" \
    uv run phase2/nlme_vi_phase2_realdata.py --dataset theoph --K 1,64 --max-steps 40000 --out "$OUT"; then
    cp "$OUT/phase2_realdata_results.csv" "$OUT/phase2_realdata_theoph.csv"
fi

if [[ "$HAVE_WARFARIN" -eq 1 ]]; then
    if run_step "Phase 2, real data: warfarin (extract PK rows)" \
        python3 -c "
import pandas as pd
df = pd.read_csv('warfarin.csv')
pk = df[df.dvid == 'cp']
pk.to_csv('warfarin_pk.csv', index=False)
print(f'{len(pk)} PK rows extracted (expect 283 for the standard dataset)')
"; then
        if run_step "Phase 2, real data: warfarin (fit)" \
            uv run phase2/nlme_vi_phase2_realdata.py --csv warfarin_pk.csv \
                --col-map "id=subject,time=time,dv=conc,amt=dose" --K 1,64 --max-steps 40000 --out "$OUT"; then
            cp "$OUT/phase2_realdata_results.csv" "$OUT/phase2_realdata_warfarin.csv"
        fi
    fi
else
    echo -e "\n>>> Phase 2, real data: warfarin -- SKIPPED (warfarin.csv not found, see note above)"
fi

# ============================================================ Phase 2: PSIS
run_step "Phase 2, PSIS/ESS diagnostic" \
    python phase2/nlme_vi_phase2_psis.py --subjects 120 --bad-steps 30 --out "$OUT" || true

# ============================================================ Phase 2: dOFV
if run_step "Phase 2, dOFV -- free posterior (n=100)" \
    $CAFFEINATE uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 100 --subjects 120 --K 64 --n-workers "$WORKERS" --out "$OUT"; then
    cp "$OUT/phase2_deltaofv_results.csv" "$OUT/phase2_deltaofv_free.csv"
fi

if run_step "Phase 2, dOFV -- amortized posterior (n=300, confirmatory)" \
    $CAFFEINATE uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 300 --subjects 120 --K 64 --posterior amortized \
        --n-workers "$WORKERS" --out "$OUT"; then
    cp "$OUT/phase2_deltaofv_results.csv" "$OUT/phase2_deltaofv_amortized_n300.csv"
fi

# ============================================================ Phase 2: FOCEI/SAEM baseline
echo -e "\n>>> Phase 2, VI vs FOCEI/SAEM baseline (optional -- skips gracefully"
echo "    if R/nlmixr2 isn't set up; see README Requirements)"
BASELINE_CSV=""
if command -v Rscript >/dev/null 2>&1 && Rscript -e 'library(nlmixr2)' >/dev/null 2>&1; then
    if run_step "Phase 2, FOCEI/SAEM baseline" \
        uv run phase2/nlme_vi_phase2_baselines.py --subjects 120 --reps 20 --out "$OUT/phase2_baselines"; then
        BASELINE_CSV="$OUT/phase2_baselines/phase2_baseline_comparison.csv"
    fi
else
    echo "R/nlmixr2 not available -- skipping. This does not affect any other"
    echo "result (see README: no core claim depends on this baseline)."
fi

# ============================================================ Tables + figures
# Deliberately UNCONDITIONAL -- always runs, regardless of what failed or
# was skipped above. make_tables.py/make_figures.py already handle missing
# input files gracefully (print [skip], move on) -- this is the whole
# point of the fault-tolerant design above: a reviewer gets every table
# and figure whose inputs succeeded, not nothing just because one
# unrelated step failed somewhere in a multi-hour run.
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
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo ""
    echo "*** ${#FAILED_STEPS[@]} step(s) FAILED and were skipped -- any table"
    echo "*** or figure depending on them will be missing above, not silently"
    echo "*** wrong. Re-run this script (or the specific underlying command"
    echo "*** from commands.sh) after investigating:"
    for s in "${FAILED_STEPS[@]}"; do echo "***   - $s"; done
else
    echo "All steps completed successfully."
fi
echo "================================================================"