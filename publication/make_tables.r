#!/usr/bin/env Rscript

# publication/make_tables.R
#
# Manuscript tables from the project's result CSVs.
#
# WHY EXPLICIT FILE PATHS FOR EVERYTHING, NOT DEFAULT FILENAMES:
#
# Several scripts write to a fixed default output filename regardless of
# which condition was run. For example, a real-data script may write the
# same filename for Theophylline and Warfarin, and the dOFV script may write
# the same filename for free and amortized posterior runs.
#
# Running a second condition can therefore overwrite the first.
#
# This script does not guess which result belongs to which condition.
# It takes an explicit path for every named condition. If a path is omitted
# or the file does not exist, the corresponding table is skipped.
#
#
# USAGE
#
# Rscript publication/make_tables.R \
#   --phase0-csv outputs/phase0_results.csv \
#   --phase1-csv outputs/phase1_results.csv \
#   --nonlinear-csv outputs/phase1_nonlinear_results.csv \
#   --theoph-csv outputs/phase2_realdata_theoph.csv \
#   --warfarin-csv outputs/phase2_realdata_warfarin.csv \
#   --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
#   --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized.csv \
#   --psis-csv outputs/phase2_psis_results.csv \
#   --baseline-csv outputs/phase2_baseline_comparison.csv \
#   --out publication/tables
#
# Any flag can be omitted. That table is skipped, not treated as an error.
#
#
# OUTPUT
#
# For each table:
#
#   1. .csv  -- machine-readable output for downstream analysis/QC
#   2. .md   -- Markdown version for manuscript use
#   3. console output
#
#
# Dependencies:
#
# install.packages(c(
#   "dplyr",
#   "tidyr",
#   "readr"
# ))


# =========================================================================
# PACKAGES
# =========================================================================

library(dplyr)
library(tidyr)
library(readr)


# =========================================================================
# COMMAND-LINE ARGUMENTS
# =========================================================================

parse_args <- function() {

  args <- commandArgs(trailingOnly = TRUE)

  defaults <- list(
    phase0_csv = NULL,
    phase1_csv = NULL,
    nonlinear_csv = NULL,
    theoph_csv = NULL,
    warfarin_csv = NULL,
    deltaofv_free_csv = NULL,
    deltaofv_amortized_csv = NULL,
    psis_csv = NULL,
    baseline_csv = NULL,
    out = "publication/tables"
  )

  flag_map <- c(
    "--phase0-csv" = "phase0_csv",
    "--phase1-csv" = "phase1_csv",
    "--nonlinear-csv" = "nonlinear_csv",
    "--theoph-csv" = "theoph_csv",
    "--warfarin-csv" = "warfarin_csv",
    "--deltaofv-free-csv" = "deltaofv_free_csv",
    "--deltaofv-amortized-csv" = "deltaofv_amortized_csv",
    "--psis-csv" = "psis_csv",
    "--baseline-csv" = "baseline_csv",
    "--out" = "out"
  )

  i <- 1

  while (i <= length(args)) {

    flag <- args[i]

    if (flag %in% names(flag_map)) {

      if (i == length(args)) {
        stop("Missing value after ", flag)
      }

      key <- flag_map[[flag]]
      defaults[[key]] <- args[i + 1]

      i <- i + 2

    } else {

      warning("Unknown argument: ", flag)
      i <- i + 1
    }
  }

  defaults
}


# =========================================================================
# UTILITIES
# =========================================================================

try_load <- function(path, label) {

  if (is.null(path) || !nzchar(path)) {

    message("")
    message("[skip] ", label, ": no path given")

    return(NULL)
  }

  if (!file.exists(path)) {

    message("")
    message("[skip] ", label, ": ", path, " not found")

    return(NULL)
  }

  df <- read_csv(
    path,
    show_col_types = FALSE
  )

  message("")
  message(
    "[loaded] ",
    label,
    ": ",
    path,
    " (",
    nrow(df),
    " rows)"
  )

  df
}


# -------------------------------------------------------------------------
# Markdown table writer
# -------------------------------------------------------------------------

df_to_markdown <- function(df, index = FALSE) {

  # R data frames do not have an explicit pandas-style index, so `index`
  # is retained only to parallel the Python function interface.
  #
  # The actual table content is written using the visible columns.

  if (ncol(df) == 0) {
    return("")
  }

  format_cell <- function(x) {

    if (is.na(x)) {
      return("NA")
    }

    x <- as.character(x)

    # Avoid breaking Markdown tables if a value contains a pipe.
    gsub("\\|", "\\\\|", x)
  }

  header <- paste0(
    "| ",
    paste(names(df), collapse = " | "),
    " |"
  )

  separator <- paste0(
    "| ",
    paste(rep("---", ncol(df)), collapse = " | "),
    " |"
  )

  rows <- apply(
    df,
    1,
    function(row) {
      paste0(
        "| ",
        paste(
          vapply(row, format_cell, character(1)),
          collapse = " | "
        ),
        " |"
      )
    }
  )

  paste(
    c(
      header,
      separator,
      rows
    ),
    collapse = "\n"
  )
}


# -------------------------------------------------------------------------
# Save table
# -------------------------------------------------------------------------

save_table <- function(
    df,
    name,
    out_dir,
    index = FALSE
) {

  dir.create(
    out_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  csv_path <- file.path(
    out_dir,
    paste0(name, ".csv")
  )

  md_path <- file.path(
    out_dir,
    paste0(name, ".md")
  )

  write_csv(
    df,
    csv_path,
    na = ""
  )

  writeLines(
    df_to_markdown(
      df,
      index = index
    ),
    md_path
  )

  cat("\n")
  cat("=== ", name, " ===\n", sep = "")

  print(
    df,
    n = Inf,
    width = Inf
  )

  cat(
    "-> ",
    csv_path,
    "\n-> ",
    md_path,
    "\n",
    sep = ""
  )
}


# =========================================================================
# TABLE BUILDERS
# =========================================================================


# -------------------------------------------------------------------------
# Phase 0
# -------------------------------------------------------------------------

table_phase0 <- function(df) {

  # Headline go/no-go:
  # omega bias by posterior x K.

  df %>%
    filter(
      startsWith(param, "om_")
    ) %>%
    group_by(
      posterior,
      K
    ) %>%
    summarise(
      mean_bias_pct = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sd_bias_pct = sd(
        rel_bias_pct,
        na.rm = TRUE
      ),
      n_estimates = sum(
        !is.na(rel_bias_pct)
      ),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(
          mean_bias_pct,
          sd_bias_pct
        ),
        ~ round(.x, 2)
      )
    )
}


# -------------------------------------------------------------------------
# Phase 1 linear grid
# -------------------------------------------------------------------------

table_phase1_grid <- function(df) {

  # Q2/Q4:
  #
  # Omega bias by
  # scenario x posterior x family x K.
  #
  # Only the dense/sparse linear scenarios are included.
  # Nonlinear is handled separately because it has different parameter
  # names and different reporting goals.

  df %>%
    filter(
      startsWith(param, "om_"),
      scenario %in% c(
        "dense",
        "sparse"
      )
    ) %>%
    group_by(
      scenario,
      posterior,
      family,
      K
    ) %>%
    summarise(
      mean_bias_pct = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sd_bias_pct = sd(
        rel_bias_pct,
        na.rm = TRUE
      ),
      n_estimates = sum(
        !is.na(rel_bias_pct)
      ),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(
          mean_bias_pct,
          sd_bias_pct
        ),
        ~ round(.x, 2)
      )
    )
}


# -------------------------------------------------------------------------
# Nonlinear tier
# -------------------------------------------------------------------------

table_nonlinear <- function(df) {

  # Q3:
  #
  # Nonlinear Michaelis-Menten tier.
  #
  # Report all parameters, not only omega parameters, because recovery
  # of fixed effects such as Vmax / Km / V is itself part of the result.

  nl <- df

  if ("scenario" %in% names(df)) {

    nl <- df %>%
      filter(
        scenario == "nonlinear"
      )
  }

  nl %>%
    group_by(
      posterior,
      family,
      K,
      param
    ) %>%
    summarise(
      mean_estimate = mean(
        estimate,
        na.rm = TRUE
      ),
      truth = first(
        truth
      ),
      mean_bias_pct = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      n = sum(
        !is.na(rel_bias_pct)
      ),
      frac_converged = mean(
        converged,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(
          mean_estimate,
          truth,
          mean_bias_pct,
          frac_converged
        ),
        ~ round(.x, 3)
      )
    )
}


# -------------------------------------------------------------------------
# Real data
# -------------------------------------------------------------------------

table_realdata <- function(
    df,
    dataset_name
) {

  # K-low vs K-high:
  # all recognized parameters for one real dataset.

  k_lo <- min(
    df$K,
    na.rm = TRUE
  )

  k_hi <- max(
    df$K,
    na.rm = TRUE
  )

  lo <- df %>%
    filter(
      K == k_lo
    ) %>%
    slice(1)

  hi <- df %>%
    filter(
      K == k_hi
    ) %>%
    slice(1)

  candidate_params <- c(
    "CL",
    "V",
    "ka",
    "om_CL",
    "om_V",
    "om_ka",
    "sigma"
  )

  param_cols <- intersect(
    candidate_params,
    names(df)
  )

  rows <- lapply(
    param_cols,
    function(p) {

      estimate_lo <- lo[[p]][1]
      estimate_hi <- hi[[p]][1]

      if (
        is.na(estimate_hi) ||
        estimate_hi == 0
      ) {

        rel <- NA_real_

      } else {

        rel <- 100 *
          (estimate_lo - estimate_hi) /
          estimate_hi
      }

      tibble(
        dataset = dataset_name,
        param = p,
        K_low = k_lo,
        estimate_K_low = estimate_lo,
        K_high = k_hi,
        estimate_K_high = estimate_hi,
        pct_diff_low_vs_high = round(
          rel,
          2
        )
      )
    }
  )

  bind_rows(rows)
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

table_deltaofv_calibration <- function(
    df,
    label
) {

  # Boundary fraction + KS test against the chi-square(1) positive
  # component of the Self-Liang reference mixture.
  #
  # The boundary mass itself is handled separately through the fraction
  # with dOFV <= 0.05.

  dofv <- df$dofv

  dofv <- dofv[
    !is.na(dofv)
  ]

  frac_boundary <- 100 *
    mean(
      dofv <= 0.05
    )

  positive <- dofv[
    dofv > 0.05
  ]

  if (length(positive) >= 10) {

    ks_result <- suppressWarnings(
      ks.test(
        positive,
        "pchisq",
        df = 1
      )
    )

    ks_stat <- unname(
      ks_result$statistic
    )

    ks_p <- ks_result$p.value

  } else {

    ks_stat <- NA_real_
    ks_p <- NA_real_
  }

  # Self-Liang:
  #
  # 0.5 * point mass at 0
  # +
  # 0.5 * chi-square(df = 1)
  #
  # A 5% upper-tail test for the mixture corresponds to the
  # 90th percentile of chi-square(1).

  correct_cutoff <- qchisq(
    0.90,
    df = 1
  )

  type1_correct <- 100 *
    mean(
      dofv > correct_cutoff
    )

  n_negative <- sum(
    dofv < 0
  )

  tibble(
    condition = label,
    n_reps = length(dofv),
    boundary_fraction_pct = round(
      frac_boundary,
      1
    ),
    n_negative_dofv = as.integer(
      n_negative
    ),
    ks_stat = round(
      ks_stat,
      4
    ),
    ks_p = round(
      ks_p,
      4
    ),
    type1_error_correct_test_pct = round(
      type1_correct,
      1
    )
  )
}


# -------------------------------------------------------------------------
# PSIS / ESS
# -------------------------------------------------------------------------

table_psis <- function(df) {

  # Aggregate ESS by arm.
  #
  # Per-subject tail-threshold metrics are intentionally not summarized
  # here, matching the Python implementation.

  df %>%
    group_by(
      arm
    ) %>%
    summarise(
      mean = mean(
        ess,
        na.rm = TRUE
      ),
      median = median(
        ess,
        na.rm = TRUE
      ),
      min = min(
        ess,
        na.rm = TRUE
      ),
      max = max(
        ess,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(
          mean,
          median,
          min,
          max
        ),
        ~ round(.x, 1)
      )
    )
}


# -------------------------------------------------------------------------
# VI vs FOCEI vs SAEM
# -------------------------------------------------------------------------

table_baseline_comparison <- function(df) {

  # VI vs FOCEI vs SAEM:
  # mean parameter estimates plus runtime.
  #
  # If truth_* columns are present, append a TRUTH row.

  candidate_params <- c(
    "CL",
    "V",
    "ka",
    "om_CL",
    "om_V",
    "om_ka",
    "sigma"
  )

  param_cols <- intersect(
    candidate_params,
    names(df)
  )

  if (length(param_cols) == 0) {
    stop(
      "Baseline comparison file contains none of the expected parameter columns."
    )
  }

  est <- df %>%
    group_by(
      method
    ) %>%
    summarise(
      across(
        all_of(param_cols),
        ~ mean(.x, na.rm = TRUE)
      ),
      .groups = "drop"
    )

  # -----------------------------------------------------------------------
  # Add runtime if available
  # -----------------------------------------------------------------------

  if ("cpu_secs" %in% names(df)) {

    runtime <- df %>%
      group_by(
        method
      ) %>%
      summarise(
        cpu_secs_mean = mean(
          cpu_secs,
          na.rm = TRUE
        ),
        cpu_secs_median = median(
          cpu_secs,
          na.rm = TRUE
        ),
        .groups = "drop"
      )

    est <- est %>%
      left_join(
        runtime,
        by = "method"
      )
  }

  # -----------------------------------------------------------------------
  # Add truth row if truth columns exist
  # -----------------------------------------------------------------------

  truth_names <- paste0(
    "truth_",
    param_cols
  )

  available_truth <- truth_names[
    truth_names %in% names(df)
  ]

  if (length(available_truth) > 0) {

    truth_row <- tibble(
      method = "TRUTH"
    )

    for (p in param_cols) {

      truth_col <- paste0(
        "truth_",
        p
      )

      if (truth_col %in% names(df)) {

        truth_row[[p]] <- df[[truth_col]][1]

      } else {

        truth_row[[p]] <- NA_real_
      }
    }

    if ("cpu_secs_mean" %in% names(est)) {
      truth_row$cpu_secs_mean <- NA_real_
    }

    if ("cpu_secs_median" %in% names(est)) {
      truth_row$cpu_secs_median <- NA_real_
    }

    truth_row <- truth_row %>%
      select(
        all_of(names(est))
      )

    est <- bind_rows(
      est,
      truth_row
    )
  }

  est %>%
    mutate(
      across(
        where(is.numeric),
        ~ round(.x, 4)
      )
    )
}


# =========================================================================
# MAIN
# =========================================================================

args <- parse_args()


cat(
  strrep("=", 72),
  "\n",
  sep = ""
)

cat(
  "MANUSCRIPT TABLES\n"
)

cat(
  strrep("=", 72),
  "\n",
  sep = ""
)


# -------------------------------------------------------------------------
# Phase 0
# -------------------------------------------------------------------------

df <- try_load(
  args$phase0_csv,
  "Phase 0 (go/no-go)"
)

if (!is.null(df)) {

  save_table(
    table_phase0(df),
    "table_phase0_headline",
    args$out
  )
}


# -------------------------------------------------------------------------
# Phase 1 linear
# -------------------------------------------------------------------------

df <- try_load(
  args$phase1_csv,
  "Phase 1 grid (Q2/Q4, linear)"
)

if (!is.null(df)) {

  save_table(
    table_phase1_grid(df),
    "table_phase1_q2q4",
    args$out
  )
}


# -------------------------------------------------------------------------
# Phase 1 nonlinear
# -------------------------------------------------------------------------

nonlinear_path <- if (!is.null(args$nonlinear_csv)) {

  args$nonlinear_csv

} else {

  args$phase1_csv
}


df <- try_load(
  nonlinear_path,
  "Phase 1 nonlinear (Q3)"
)

if (!is.null(df)) {

  has_nonlinear <-
    "scenario" %in% names(df) &&
    "nonlinear" %in% df$scenario

  if (has_nonlinear) {

    save_table(
      table_nonlinear(df),
      "table_phase1_q3_nonlinear",
      args$out
    )

  } else {

    message(
      "[skip] Q3 nonlinear table: ",
      "no 'nonlinear' scenario rows in the given file"
    )
  }
}


# -------------------------------------------------------------------------
# Real data
# -------------------------------------------------------------------------

realdata_tables <- list()


df <- try_load(
  args$theoph_csv,
  "Real data: Theophylline"
)

if (!is.null(df)) {

  realdata_tables[[
    length(realdata_tables) + 1
  ]] <- table_realdata(
    df,
    "theophylline"
  )
}


df <- try_load(
  args$warfarin_csv,
  "Real data: warfarin"
)

if (!is.null(df)) {

  realdata_tables[[
    length(realdata_tables) + 1
  ]] <- table_realdata(
    df,
    "warfarin"
  )
}


if (length(realdata_tables) > 0) {

  save_table(
    bind_rows(
      realdata_tables
    ),
    "table_realdata_k1_vs_khigh",
    args$out
  )
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

calib_tables <- list()


df <- try_load(
  args$deltaofv_free_csv,
  "dOFV calibration: free posterior"
)

if (!is.null(df)) {

  calib_tables[[
    length(calib_tables) + 1
  ]] <- table_deltaofv_calibration(
    df,
    "free"
  )
}


df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV calibration: amortized posterior"
)

if (!is.null(df)) {

  calib_tables[[
    length(calib_tables) + 1
  ]] <- table_deltaofv_calibration(
    df,
    "amortized"
  )
}


if (length(calib_tables) > 0) {

  save_table(
    bind_rows(
      calib_tables
    ),
    "table_deltaofv_calibration",
    args$out
  )
}


# -------------------------------------------------------------------------
# PSIS / ESS
# -------------------------------------------------------------------------

df <- try_load(
  args$psis_csv,
  "PSIS/ESS diagnostic"
)

if (!is.null(df)) {

  save_table(
    table_psis(df),
    "table_psis_ess",
    args$out
  )
}


# -------------------------------------------------------------------------
# VI vs FOCEI vs SAEM
# -------------------------------------------------------------------------

df <- try_load(
  args$baseline_csv,
  "VI vs FOCEI vs SAEM"
)

if (!is.null(df)) {

  save_table(
    table_baseline_comparison(df),
    "table_baseline_comparison",
    args$out
  )
}


# -------------------------------------------------------------------------
# Finish
# -------------------------------------------------------------------------

cat("\n")
cat(strrep("=", 72), "\n", sep = "")

cat(
  "Done. Tables written to ",
  args$out,
  "/\n",
  sep = ""
)

cat(
  "Skipped tables simply mean that the corresponding source file was\n",
  "not supplied or was not found. Rerun with additional flags as more\n",
  "results become available.\n",
  sep = ""
)