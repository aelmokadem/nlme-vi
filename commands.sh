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

phase1_nonlinear() {
    # Expensive (RK4-integrated, no closed form) -- kept small by default.
    python nlme_vi_phase1.py --scenarios dense,sparse,nonlinear \
        --nl-reps 20 --nl-steps 1500 --nl-subjects 60
}

# ===================================================== Phase 2: real data
realdata_theoph() {
    uv run phase2/nlme_vi_phase2_realdata.py --dataset theoph --K 1,64 --max-steps 40000
}

realdata_warfarin() {
    # Requires warfarin.csv already exported from R -- see README, no
    # compilation needed for the export itself (nlmixr2data is a pure
    # data package).
    local csv="${1:-warfarin.csv}"
    if [[ ! -f "$csv" ]]; then
        echo "Missing $csv. In R first:"
        echo '  install.packages("nlmixr2data")'
        echo "  write.csv(nlmixr2data::warfarin, \"$csv\", row.names = FALSE)"
        echo "Then check column names with: head -3 $csv"
        return 1
    fi
    uv run phase2/nlme_vi_phase2_realdata.py --csv "$csv" \
        --col-map "ID=subject,TIME=time,DV=conc,AMT=dose" --K 1,64 --max-steps 40000
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
}

deltaofv_amortized() {
    # The fix -- compare boundary fraction / KS p-value against deltaofv_free.
    local workers; workers=$(( $(n_cores) - 1 ))
    $(maybe_caffeinate) uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 100 --subjects 120 --K 64 --n-workers "$workers" --posterior amortized
}

deltaofv_amortized_confirmatory() {
    # More reps for a decisively-powered KS test.
    local workers; workers=$(( $(n_cores) - 1 ))
    $(maybe_caffeinate) uv run phase2/nlme_vi_phase2_deltaofv.py \
        --reps 300 --subjects 120 --K 64 --n-workers "$workers" --posterior amortized
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
