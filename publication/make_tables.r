#!/usr/bin/env Rscript

# =============================================================================
# publication/make_tables.R
#
# Manuscript tables from already-generated project result CSVs.
#
# IMPORTANT:
#   This is a PURE POST-PROCESSING script.
#
#   It does NOT:
#     - run any model
#     - run nlmixr2
#     - run Python
#     - run simulations
#     - source phase scripts
#     - modify source result CSVs
#
#   It ONLY:
#     1. reads explicitly supplied CSV files
#     2. summarizes those results
#     3. writes manuscript tables
#
# WHY EXPLICIT FILE PATHS:
#   Several analysis scripts can write to the same default filename under
#   different conditions. This script never guesses which condition a file
#   represents. Each named condition must be supplied explicitly.
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
# Any input flag may be omitted. Its corresponding table is skipped.
#
# OUTPUT
#   Each table is written as:
#       .csv
#       .md
#
#   Tables are also printed to the console.
# =============================================================================


suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})


# =============================================================================
# Command-line arguments
# =============================================================================

parse_args <- function(args) {

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

  out <- defaults

  i <- 1

  while (i <= length(args)) {

    flag <- args[[i]]

    if (!flag %in% names(flag_map)) {
      stop("Unknown argument: ", flag)
    }

    if (i == length(args)) {
      stop("Missing value after ", flag)
    }

    key <- unname(flag_map[[flag]])
    out[[key]] <- args[[i + 1]]

    i <- i + 2
  }

  out
}


args <- parse_args(commandArgs(trailingOnly = TRUE))


# =============================================================================
# Utilities
# =============================================================================

try_load <- function(path, label) {

  if (is.null(path) || !nzchar(path)) {
    cat(sprintf("\n[skip] %s: no path given\n", label))
    return(NULL)
  }

  if (!file.exists(path)) {
    cat(sprintf("\n[skip] %s: %s not found\n", label, path))
    return(NULL)
  }

  df <- read_csv(
    path,
    show_col_types = FALSE
  )

  cat(
    sprintf(
      "\n[loaded] %s: %s (%d rows)\n",
      label,
      path,
      nrow(df)
    )
  )

  df
}


format_md_value <- function(x) {

  if (length(x) == 0 || is.na(x)) {
    return("")
  }

  if (is.numeric(x)) {
    return(format(x, trim = TRUE, scientific = FALSE))
  }

  as.character(x)
}


df_to_markdown <- function(df) {

  cols <- names(df)

  header <- paste0(
    "| ",
    paste(cols, collapse = " | "),
    " |"
  )

  separator <- paste0(
    "| ",
    paste(rep("---", length(cols)), collapse = " | "),
    " |"
  )

  rows <- apply(
    df,
    1,
    function(row) {
      paste0(
        "| ",
        paste(
          vapply(
            row,
            format_md_value,
            FUN.VALUE = character(1)
          ),
          collapse = " | "
        ),
        " |"
      )
    }
  )

  paste(
    c(header, separator, rows),
    collapse = "\n"
  )
}


save_table <- function(
  df,
  name,
  out_dir
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
    csv_path
  )

  writeLines(
    df_to_markdown(df),
    md_path
  )

  cat(
    sprintf(
      "\n=== %s ===\n",
      name
    )
  )

  print(
    df,
    n = Inf,
    width = Inf
  )

  cat(
    sprintf(
      "-> %s\n-> %s\n",
      csv_path,
      md_path
    )
  )
}


# =============================================================================
# Table builders
# =============================================================================

table_phase0 <- function(df) {

  df %>%
    filter(
      startsWith(param, "om_")
    ) %>%
    group_by(
      posterior,
      K,
      param
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
      mean_bias_pct = round(mean_bias_pct, 2),
      sd_bias_pct = round(sd_bias_pct, 2)
    )
}


table_phase0_fixed_effects <- function(df) {

  df %>%
    filter(
      !startsWith(param, "om_")
    ) %>%
    group_by(
      posterior,
      K,
      param
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
      mean_bias_pct = round(mean_bias_pct, 2),
      sd_bias_pct = round(sd_bias_pct, 2)
    )
}


table_phase1_grid <- function(df) {

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
      K,
      param
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
      mean_bias_pct = round(mean_bias_pct, 2),
      sd_bias_pct = round(sd_bias_pct, 2)
    )
}


table_phase1_grid_fixed_effects <- function(df) {

  df %>%
    filter(
      !startsWith(param, "om_"),
      scenario %in% c(
        "dense",
        "sparse"
      )
    ) %>%
    group_by(
      scenario,
      posterior,
      family,
      K,
      param
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
      mean_bias_pct = round(mean_bias_pct, 2),
      sd_bias_pct = round(sd_bias_pct, 2)
    )
}


table_nonlinear <- function(df) {

  if ("scenario" %in% names(df)) {
    df <- df %>%
      filter(
        scenario == "nonlinear"
      )
  }

  df %>%
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
      truth = first(truth),
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


table_realdata <- function(
  df,
  dataset_name
) {

  k_lo <- min(
    df$K,
    na.rm = TRUE
  )

  k_hi <- max(
    df$K,
    na.rm = TRUE
  )

  lo <- df %>%
    filter(K == k_lo) %>%
    slice(1)

  hi <- df %>%
    filter(K == k_hi) %>%
    slice(1)

  param_cols <- intersect(
    c(
      "CL",
      "V",
      "ka",
      "om_CL",
      "om_V",
      "om_ka",
      "sigma"
    ),
    names(df)
  )

  bind_rows(
    lapply(
      param_cols,
      function(p) {

        lo_val <- lo[[p]][[1]]
        hi_val <- hi[[p]][[1]]

        pct_diff <- if (
          is.finite(hi_val) &&
          hi_val != 0
        ) {
          100 * (lo_val - hi_val) / hi_val
        } else {
          NA_real_
        }

        tibble(
          dataset = dataset_name,
          param = p,
          K_low = k_lo,
          estimate_K_low = lo_val,
          K_high = k_hi,
          estimate_K_high = hi_val,
          pct_diff_low_vs_high = round(
            pct_diff,
            2
          )
        )
      }
    )
  )
}


table_deltaofv_calibration <- function(
  df,
  label
) {

  dofv <- df$dofv

  dofv <- dofv[
    is.finite(dofv)
  ]

  frac_boundary <- 100 * mean(
    dofv <= 0.05
  )

  positive <- dofv[
    dofv > 0.05
  ]

  if (length(positive) >= 10) {

    ks <- suppressWarnings(
      ks.test(
        positive,
        "pchisq",
        df = 1
      )
    )

    ks_stat <- unname(
      ks$statistic
    )

    ks_p <- ks$p.value

  } else {

    ks_stat <- NA_real_
    ks_p <- NA_real_
  }

  # Self-Liang 50:50 mixture:
  # alpha = 0.05 corresponds to the 90th percentile of chi-square(1).
  correct_cutoff <- qchisq(
    0.90,
    df = 1
  )

  type1_correct <- 100 * mean(
    dofv > correct_cutoff
  )

  tibble(
    condition = label,
    n_reps = length(dofv),
    boundary_fraction_pct = round(
      frac_boundary,
      1
    ),
    n_negative_dofv = sum(
      dofv < 0
    ),
    ks_stat = ifelse(
      is.na(ks_stat),
      NA_real_,
      round(ks_stat, 4)
    ),
    ks_p = ifelse(
      is.na(ks_p),
      NA_real_,
      round(ks_p, 4)
    ),
    type1_error_correct_test_pct = round(
      type1_correct,
      1
    )
  )
}


table_psis <- function(df) {

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


table_baseline_comparison <- function(df) {

  param_cols <- intersect(
    c(
      "CL",
      "V",
      "ka",
      "om_CL",
      "om_V",
      "om_ka",
      "sigma"
    ),
    names(df)
  )

  result <- df %>%
    group_by(
      method
    ) %>%
    summarise(
      across(
        all_of(param_cols),
        ~ mean(
          .x,
          na.rm = TRUE
        )
      ),
      .groups = "drop"
    )

  truth_cols <- paste0(
    "truth_",
    param_cols
  )

  truth_cols <- truth_cols[
    truth_cols %in% names(df)
  ]

  if (length(truth_cols) > 0) {

    truth <- tibble(
      method = "TRUTH"
    )

    for (tc in truth_cols) {

      p <- sub(
        "^truth_",
        "",
        tc
      )

      truth[[p]] <- df[[tc]][[1]]
    }

    result <- bind_rows(
      result,
      truth
    )
  }

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

    result <- result %>%
      left_join(
        runtime,
        by = "method"
      )
  }

  result %>%
    mutate(
      across(
        where(is.numeric),
        ~ round(.x, 4)
      )
    )
}


# =============================================================================
# Main
# =============================================================================

cat(
  paste0(
    "\n",
    paste(rep("=", 72), collapse = ""),
    "\nMANUSCRIPT TABLES\n",
    paste(rep("=", 72), collapse = ""),
    "\n"
  )
)


# Phase 0
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

  save_table(
    table_phase0_fixed_effects(df),
    "table_phase0_fixed_effects",
    args$out
  )
}


# Phase 1 linear grid
df_phase1 <- try_load(
  args$phase1_csv,
  "Phase 1 grid (Q2/Q4, linear)"
)

if (!is.null(df_phase1)) {

  save_table(
    table_phase1_grid(df_phase1),
    "table_phase1_q2q4",
    args$out
  )

  save_table(
    table_phase1_grid_fixed_effects(df_phase1),
    "table_phase1_q2q4_fixed_effects",
    args$out
  )
}


# Nonlinear
nonlinear_path <- if (
  !is.null(args$nonlinear_csv)
) {
  args$nonlinear_csv
} else {
  args$phase1_csv
}

df_nonlinear <- try_load(
  nonlinear_path,
  "Phase 1 nonlinear (Q3)"
)

if (!is.null(df_nonlinear)) {

  has_nonlinear <- if (
    "scenario" %in% names(df_nonlinear)
  ) {
    any(
      df_nonlinear$scenario == "nonlinear",
      na.rm = TRUE
    )
  } else {
    TRUE
  }

  if (has_nonlinear) {

    save_table(
      table_nonlinear(df_nonlinear),
      "table_phase1_q3_nonlinear",
      args$out
    )

  } else {

    cat(
      "[skip] Q3 nonlinear table: ",
      "no 'nonlinear' scenario rows in the given file\n",
      sep = ""
    )
  }
}


# Real data
realdata_tables <- list()

df <- try_load(
  args$theoph_csv,
  "Real data: Theophylline"
)

if (!is.null(df)) {

  realdata_tables[[length(realdata_tables) + 1]] <-
    table_realdata(
      df,
      "theophylline"
    )
}

df <- try_load(
  args$warfarin_csv,
  "Real data: warfarin"
)

if (!is.null(df)) {

  realdata_tables[[length(realdata_tables) + 1]] <-
    table_realdata(
      df,
      "warfarin"
    )
}

if (length(realdata_tables) > 0) {

  save_table(
    bind_rows(realdata_tables),
    "table_realdata_k1_vs_khigh",
    args$out
  )
}


# dOFV calibration
calibration_tables <- list()

df <- try_load(
  args$deltaofv_free_csv,
  "dOFV calibration: free posterior"
)

if (!is.null(df)) {

  calibration_tables[[length(calibration_tables) + 1]] <-
    table_deltaofv_calibration(
      df,
      "free"
    )
}

df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV calibration: amortized posterior"
)

if (!is.null(df)) {

  calibration_tables[[length(calibration_tables) + 1]] <-
    table_deltaofv_calibration(
      df,
      "amortized"
    )
}

if (length(calibration_tables) > 0) {

  save_table(
    bind_rows(calibration_tables),
    "table_deltaofv_calibration",
    args$out
  )
}


# PSIS / ESS
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


# VI vs FOCEI vs SAEM
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


cat(
  paste0(
    "\n",
    paste(rep("=", 72), collapse = ""),
    "\nDone. Tables written to ",
    args$out,
    "/\n",
    "Skipped tables simply mean their source file was not supplied or found.\n"
  )
)