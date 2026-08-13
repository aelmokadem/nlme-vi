#!/usr/bin/env bash
# commands.sh -- canonical run commands, one function per task.
#
# NOT a pipeline. Every function here is independent and safe to run alone.
# Given how iterative this project is (tune a flag, look at the result,
# decide what to run next), a single "run everything" script would work
# against the actual workflow -- this is a reference/cheat-sheet you call
# into, not something meant to run start-to-finish unattended.
#
# Usage:
#   bash commands.sh                    # list what's available
#   bash commands.sh <function_name>    # run exactly that one
#
# Or just open this file and copy the line you want -- nothing here depends
# on being invoked through this dispatcher.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"   # always run from the repo root

# --------------------------------------------------------------- helpers
n_cores() {
    if command -v nproc >/dev/null 2>&1; then nproc
    elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu
    else echo 4
    fi
}

has_caffeinate() { command -v caffeinate >/dev/null 2>&1; }
maybe_caffeinate() { if has_caffeinate; then echo "caffeinate -i"; fi }

# ===================================================== Phase 0
phase0_quick() {
    python nlme_vi_phase0.py --quick
}

phase0_full() {
    python nlme_vi_phase0.py --reps 8 --subjects 120 --max-steps 25000
}

# ===================================================== Phase 1
phase1_converge_check() {
    # Run this before trusting any bias number from anywhere in the repo.
    python nlme_vi_phase1.py --converge-check --converge-steps 25000
}

phase1_gaussian_grid() {
    # Both posteriors stable under gaussian.
    python nlme_vi_phase1.py --scenarios dense,sparse --families gaussian \
        --posteriors free,amortized --K 1,8,64 --reps 30
}

phase1_flow_grid() {
    # FREE POSTERIOR ONLY -- amortized+flow is unstable, see README Known Issues.
    python nlme_vi_phase1.py --scenarios dense,sparse --families flow \
        --posteriors free --K 1,8,64 --reps 30
}

phase1_nonlinear_check() {
    # Cheap exploratory scale -- confirms the pipeline runs and gives a
    # rough directional read before committing to the expensive production
    # run below. NOT a production result.
    python nlme_vi_phase1.py --scenarios nonlinear --families gaussian \
        --posteriors free --K 1,64 --nl-reps 1 --nl-subjects 15 --mm-dt 0.5 \
        --max-steps 9000 --device cpu --out .
}

phase1_nonlinear_production() {
    # The real run. Model/design are finalized (see README Key Finding 6:
    # OneCmtIVBolusMMNoKmRE, dose=300, 60h window -- baked into
    # get_scenario(), not exposed as flags here). free+gaussian ONLY --
    # amortized+gaussian was tested at full scale and failed
    # catastrophically on this tier (bias +230% to +1168%, worsening with
    # K; a NEW instability, distinct from amortized+flow). flow was never
    # tested here and is excluded for the same reason as always (unproven
    # + amortized+flow's known instability elsewhere makes any amortized
    # combination worth extra caution on this tier specifically).
    local workers; workers=$(( $(n_cores) - 1 ))
    $(maybe_caffeinate) python nlme_vi_phase1.py --scenarios nonlinear \
        --families gaussian --posteriors free \
        --nl-reps 20 --nl-subjects 60 --mm-dt 0.1 --n-workers "$workers" --out .
    echo ""
    echo "IMPORTANT: copy phase1_results.csv to a distinct name now, e.g.:"
    echo "  cp phase1_results.csv phase1_nonlinear_results.csv"
    echo "before running anything else that writes to the same default filename."
}

# ===================================================== Phase 2: real data
realdata_theoph() {
    uv run phase2/nlme_vi_phase2_realdata.py --dataset theoph --K 1,64 --max-steps 40000
    cp outputs/phase2_realdata_results.csv outputs/phase2_realdata_theoph.csv 2>/dev/null \
        || cp phase2_realdata_results.csv phase2_realdata_theoph.csv
    echo "Saved -> phase2_realdata_theoph.csv (CSV-overwrite gotcha -- see README)"
}

realdata_warfarin() {
    # Requires warfarin.csv already exported from R -- see README, no
    # compilation needed for the export itself (nlmixr2data is a pure
    # data package). Real columns are LOWERCASE (id/time/dv/amt/dvid/evid),
    # confirmed against nlmixr2's own docs -- NOT uppercase ID/TIME/DV/AMT.
    # Also requires filtering to dvid=='cp' (the PK endpoint) FIRST --
    # warfarin is a multi-endpoint file (283 PK rows + 232 PD/'pca' rows in
    # the same table); skipping this silently mixes PK and PD observations
    # into one "concentration" column.
    local csv="${1:-warfarin.csv}"
    if [[ ! -f "$csv" ]]; then
        echo "Missing $csv. In R first:"
        echo '  install.packages("nlmixr2data")'
        echo "  write.csv(nlmixr2data::warfarin, \"$csv\", row.names = FALSE)"
        echo "Then check column names with: head -3 $csv"
        return 1
    fi
    local pk_csv="${csv%.csv}_pk.csv"
    python3 -c "
import pandas as pd
df = pd.read_csv('$csv')
pk = df[df.dvid == 'cp']
pk.to_csv('$pk_csv', index=False)
print(f'{len(pk)} PK rows extracted -> $pk_csv (expect 283 for the standard dataset)')
"
    uv run phase2/nlme_vi_phase2_realdata.py --csv "$pk_csv" \
        --col-map "id=subject,time=time,dv=conc,amt=dose" --K 1,64 --max-steps 40000
    cp outputs/phase2_realdata_results.csv outputs/phase2_realdata_warfarin.csv 2>/dev/null \
        || cp phase2_realdata_results.csv phase2_realdata_warfarin.csv
    echo "Saved -> phase2_realdata_warfarin.csv (CSV-overwrite gotcha -- see README)"
}

realdata_nscaling() {
    # Tests whether small-N residual bias is classical MLE bias (shrinks
    # with N) vs a VI-specific failure (would not shrink with N).
    local n="${1:-40}"
    uv run phase2/nlme_vi_phase2_realdata.py --n-subj "$n" --K 1,64
}

# ===================================================== Phase 2: PSIS
psis_check() {
    python phase2/nlme_vi_phase2_psis.py --subjects 120 --bad-steps 30
}

# ===================================================== Phase 2: dOFV
deltaofv_free() {
    # Reproduces the miscalibration finding (Key Finding 5 in README).
    local workers; workers=$(( $(n_cores) - 1 ))
    $(maybe_caffeinate) uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 100 --subjects 120 --K 64 --n-workers "$workers"
    cp outputs/phase2_deltaofv_results.csv outputs/phase2_deltaofv_free.csv 2>/dev/null \
        || cp phase2_deltaofv_results.csv phase2_deltaofv_free.csv
    echo "Saved -> phase2_deltaofv_free.csv (CSV-overwrite gotcha -- see README)"
}

deltaofv_amortized() {
    # The fix -- compare boundary fraction / KS p-value against deltaofv_free.
    local workers; workers=$(( $(n_cores) - 1 ))
    $(maybe_caffeinate) uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 100 --subjects 120 --K 64 --n-workers "$workers" --posterior amortized
    cp outputs/phase2_deltaofv_results.csv outputs/phase2_deltaofv_amortized.csv 2>/dev/null \
        || cp phase2_deltaofv_results.csv phase2_deltaofv_amortized.csv
    echo "Saved -> phase2_deltaofv_amortized.csv (CSV-overwrite gotcha -- see README)"
}

deltaofv_amortized_confirmatory() {
    # More reps for a decisively-powered KS test. Note: at n=300 the KS
    # p-value looks WORSE than n=100 (statistical power, not a regression --
    # see README's deltaofv section for why the KS *statistic*, not the
    # p-value, is the number to compare across sample sizes).
    local workers; workers=$(( $(n_cores) - 1 ))
    $(maybe_caffeinate) uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 300 --subjects 120 --K 64 --n-workers "$workers" --posterior amortized
    cp outputs/phase2_deltaofv_results.csv outputs/phase2_deltaofv_amortized_n300.csv 2>/dev/null \
        || cp phase2_deltaofv_results.csv phase2_deltaofv_amortized_n300.csv
    echo "Saved -> phase2_deltaofv_amortized_n300.csv (CSV-overwrite gotcha -- see README)"
}

deltaofv_evalk_test() {
    # The (ruled-out) IS-evaluation-noise hypothesis. Kept for reference/
    # reproducibility -- eval-k was NOT the explanation, --posterior was.
    local workers; workers=$(( $(n_cores) - 1 ))
    $(maybe_caffeinate) uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 20 --subjects 120 --K 64 --eval-k 20000 --n-workers "$workers"
}

# ===================================================== Phase 2: baselines
baselines_dryrun() {
    # No R needed -- validates the Python-side pipeline only.
    uv run phase2/nlme_vi_phase2_baselines.py --subjects 20 --reps 1 --dry-run
}

baselines_real() {
    # Requires a working R/rxode2 environment -- see README Requirements.
    uv run phase2/nlme_vi_phase2_baselines.py --subjects 20 --reps 1
}

baselines_r_direct() {
    # Bypasses the Python orchestrator entirely -- useful for isolating R
    # issues without the Python subprocess layer in the way.
    local csv="${1:-outputs/phase2_baselines/rep0_data.csv}"
    if [[ ! -f "$csv" ]]; then
        echo "Missing $csv -- run baselines_dryrun first to generate it."
        return 1
    fi
    Rscript phase2/baseline_nlmixr2.R "$csv" /tmp/test_foce.csv foce
}

# ===================================================== Publication
publication_tables() {
    # Every flag is optional -- run with whatever CSVs you have saved so
    # far (see the CSV-overwrite gotcha in the functions above); missing
    # sources are skipped, not an error.
    python publication/make_tables.py \
        --phase0-csv outputs/phase0_results.csv \
        --phase1-csv outputs/phase1_results.csv \
        --nonlinear-csv outputs/phase1_nonlinear_results.csv \
        --theoph-csv outputs/phase2_realdata_theoph.csv \
        --warfarin-csv outputs/phase2_realdata_warfarin.csv \
        --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
        --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized_n300.csv \
        --psis-csv outputs/phase2_psis_results.csv \
        --baseline-csv outputs/phase2_baseline_comparison.csv \
        --out publication/tables
}

publication_figures() {
    python publication/make_figures.py \
        --phase0-csv outputs/phase0_results.csv \
        --phase1-csv outputs/phase1_results.csv \
        --nonlinear-csv outputs/phase1_nonlinear_results.csv \
        --theoph-csv outputs/phase2_realdata_theoph.csv \
        --warfarin-csv outputs/phase2_realdata_warfarin.csv \
        --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
        --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized_n300.csv \
        --psis-csv outputs/phase2_psis_results.csv \
        --out publication/figures
}

# --------------------------------------------------------------- dispatch
# Only dispatch when EXECUTED directly (bash commands.sh ...), not when
# SOURCED (source commands.sh). Without this check, sourcing would hit the
# same $#-based logic below and either print the listing or `exit 0` --
# exiting the calling shell/session rather than just defining the
# functions and returning control, which defeats the documented
# `source commands.sh` usage mode entirely.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -eq 0 ]]; then
        echo "commands.sh -- available functions:"
        echo ""
        grep -E '^[a-z0-9_]+\(\) \{' "${BASH_SOURCE[0]}" \
            | sed 's/() {.*//' \
            | grep -vE '^(n_cores|has_caffeinate|maybe_caffeinate)$' \
            | sed 's/^/  /'
        echo ""
        echo "Usage: bash commands.sh <function_name>"
        echo "   or: source commands.sh   (then call functions directly in your shell)"
        exit 0
    fi
    "$@"
fi
