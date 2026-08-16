#!/usr/bin/env Rscript
# =============================================================================
# publication/make_tables.R
#
# Manuscript tables from the project's result CSVs.
#
# Explicit paths are used for every condition because several source scripts
# can write to the same default output filename. This script never guesses
# which condition a result file represents.
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
# Any flag may be omitted. Its corresponding table is skipped.
#
# Each table is written as:
#   *.csv
#   *.md
#
# and printed to the console.
# =============================================================================


# -----------------------------------------------------------------------------
# Packages
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(optparse)
  library(here)
  library(fs)
})


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

resolve_path <- function(path) {

  if (is.null(path) || is.na(path) || path == "") {
    return(NULL)
  }

  if (fs::is_absolute_path(path)) {
    path
  } else {
    here::here(path)
  }
}


try_load <- function(path, label) {

  if (is.null(path) || is.na(path) || path == "") {
    message("\n[skip] ", label, ": no path given")
    return(NULL)
  }

  full_path <- resolve_path(path)

  if (!file.exists(full_path)) {
    message("\n[skip] ", label, ": ", full_path, " not found")
    return(NULL)
  }

  df <- readr::read_csv(
    full_path,
    show_col_types = FALSE,
    na = c("", "NA", "NaN", ".")
  )

  message(
    "\n[loaded] ", label,
    ": ", full_path,
    " (", nrow(df), " rows)"
  )

  df
}


# Minimal markdown writer so we do not depend on knitr/kableExtra/etc.
df_to_markdown <- function(df) {

  df_chr <- df %>%
    mutate(
      across(
        everything(),
        ~ ifelse(is.na(.x), "NA", as.character(.x))
      )
    )

  header <- paste0(
    "| ",
    paste(names(df_chr), collapse = " | "),
    " |"
  )

  divider <- paste0(
    "| ",
    paste(rep("---", ncol(df_chr)), collapse = " | "),
    " |"
  )

  body <- apply(
    df_chr,
    1,
    function(row) {
      paste0(
        "| ",
        paste(row, collapse = " | "),
        " |"
      )
    }
  )

  paste(
    c(
      header,
      divider,
      body
    ),
    collapse = "\n"
  )
}


save_table <- function(df, name, out_dir) {

  full_out_dir <- resolve_path(out_dir)

  dir.create(
    full_out_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  csv_path <- file.path(
    full_out_dir,
    paste0(name, ".csv")
  )

  md_path <- file.path(
    full_out_dir,
    paste0(name, ".md")
  )

  readr::write_csv(
    df,
    csv_path,
    na = "NA"
  )

  writeLines(
    df_to_markdown(df),
    md_path
  )

  cat(
    "\n=== ",
    name,
    " ===\n",
    sep = ""
  )

  print(
    df,
    n = Inf,
    width = Inf
  )

  cat(
    "-> ", csv_path, "\n",
    "-> ", md_path, "\n",
    sep = ""
  )
}


# -----------------------------------------------------------------------------
# Phase 0: omega parameters
# -----------------------------------------------------------------------------

table_phase0 <- function(df) {

  df %>%
    filter(
      str_starts(param, "om_")
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
      across(
        c(
          mean_bias_pct,
          sd_bias_pct
        ),
        ~ round(.x, 2)
      )
    ) %>%
    arrange(
      posterior,
      K,
      param
    )
}


# -----------------------------------------------------------------------------
# Phase 0: fixed effects
# -----------------------------------------------------------------------------

table_phase0_fixed_effects <- function(df) {

  df %>%
    filter(
      !str_starts(param, "om_")
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
      across(
        c(
          mean_bias_pct,
          sd_bias_pct
        ),
        ~ round(.x, 2)
      )
    ) %>%
    arrange(
      posterior,
      K,
      param
    )
}


# -----------------------------------------------------------------------------
# Phase 1 grid: omega parameters
# -----------------------------------------------------------------------------

table_phase1_grid <- function(df) {

  df %>%
    filter(
      str_starts(param, "om_"),
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
      across(
        c(
          mean_bias_pct,
          sd_bias_pct
        ),
        ~ round(.x, 2)
      )
    ) %>%
    arrange(
      scenario,
      family,
      posterior,
      K,
      param
    )
}


# -----------------------------------------------------------------------------
# Phase 1 grid: fixed effects
# -----------------------------------------------------------------------------

table_phase1_grid_fixed_effects <- function(df) {

  df %>%
    filter(
      !str_starts(param, "om_"),
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
      across(
        c(
          mean_bias_pct,
          sd_bias_pct
        ),
        ~ round(.x, 2)
      )
    ) %>%
    arrange(
      scenario,
      family,
      posterior,
      K,
      param
    )
}


# -----------------------------------------------------------------------------
# Phase 1 nonlinear
# -----------------------------------------------------------------------------

table_nonlinear <- function(df) {

  nl <- if ("scenario" %in% names(df)) {

    df %>%
      filter(
        scenario == "nonlinear"
      )

  } else {

    df
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
        where(is.numeric),
        ~ round(.x, 3)
      )
    ) %>%
    arrange(
      family,
      posterior,
      K,
      param
    )
}


# -----------------------------------------------------------------------------
# Real-data comparison
# -----------------------------------------------------------------------------

table_realdata <- function(df, dataset_name) {

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

  purrr::map_dfr(
    param_cols,
    function(p) {

      lo_value <- lo[[p]][1]
      hi_value <- hi[[p]][1]

      pct_diff <- if (
        !is.na(hi_value) &&
        hi_value != 0
      ) {

        100 *
          (lo_value - hi_value) /
          hi_value

      } else {

        NA_real_
      }

      tibble(
        dataset = dataset_name,
        param = p,
        K_low = k_lo,
        estimate_K_low = lo_value,
        K_high = k_hi,
        estimate_K_high = hi_value,
        pct_diff_low_vs_high = round(
          pct_diff,
          2
        )
      )
    }
  )
}


# -----------------------------------------------------------------------------
# dOFV calibration
# -----------------------------------------------------------------------------

table_deltaofv_calibration <- function(df, label) {

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

  # Same idea as:
  # scipy.stats.kstest(positive, "chi2", args=(1,))
  #
  # Test positive dOFV values against chi-square(1).
  if (length(positive) >= 10) {

    ks_result <- suppressWarnings(
      stats::ks.test(
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

  # Self-Liang 50:50 mixture:
  # 95th percentile of mixture corresponds to 90th percentile chi-square(1)
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
    n_negative_dofv = n_negative,
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


# -----------------------------------------------------------------------------
# PSIS ESS
# -----------------------------------------------------------------------------

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
        where(is.numeric),
        ~ round(.x, 1)
      )
    )
}


# -----------------------------------------------------------------------------
# VI vs FOCEI vs SAEM
# -----------------------------------------------------------------------------

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

  if (!"method" %in% names(df)) {
    stop(
      "Baseline comparison file must contain a 'method' column."
    )
  }


  # ---------------------------------------------------------------------------
  # Standardize runtime column
  #
  # Preferred name:
  #     runtime_s
  #
  # Legacy Python results may instead have:
  #     cpu_secs
  #
  # The corrected nlmixr2 script writes runtime_s.
  # ---------------------------------------------------------------------------

  if ("runtime_s" %in% names(df)) {

    df <- df %>%
      mutate(
        runtime_s_std = runtime_s
      )

    message(
      "Baseline comparison: using 'runtime_s' for runtime."
    )

  } else if ("cpu_secs" %in% names(df)) {

    df <- df %>%
      mutate(
        runtime_s_std = cpu_secs
      )

    message(
      "Baseline comparison: using legacy 'cpu_secs' as runtime."
    )

  } else {

    df <- df %>%
      mutate(
        runtime_s_std = NA_real_
      )

    warning(
      "Baseline comparison contains neither 'runtime_s' nor 'cpu_secs'. ",
      "Runtime columns will be NA."
    )
  }


  # ---------------------------------------------------------------------------
  # Parameter estimates + runtime
  # ---------------------------------------------------------------------------

  est <- df %>%
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

      runtime_s_mean = if (
        all(is.na(runtime_s_std))
      ) {
        NA_real_
      } else {
        mean(
          runtime_s_std,
          na.rm = TRUE
        )
      },

      runtime_s_median = if (
        all(is.na(runtime_s_std))
      ) {
        NA_real_
      } else {
        median(
          runtime_s_std,
          na.rm = TRUE
        )
      },

      .groups = "drop"
    )


  # ---------------------------------------------------------------------------
  # Add TRUTH row when truth_* columns are available
  # ---------------------------------------------------------------------------

  truth_cols <- paste0(
    "truth_",
    param_cols
  )

  truth_cols <- truth_cols[
    truth_cols %in% names(df)
  ]

  if (length(truth_cols) > 0) {

    truth_values <- df %>%
      slice(1) %>%
      select(
        all_of(truth_cols)
      )

    names(truth_values) <- str_remove(
      names(truth_values),
      "^truth_"
    )

    # Ensure all parameter columns exist in truth row.
    for (p in param_cols) {

      if (!p %in% names(truth_values)) {
        truth_values[[p]] <- NA_real_
      }
    }

    truth_row <- truth_values %>%
      select(
        all_of(param_cols)
      ) %>%
      mutate(
        method = "TRUTH",
        runtime_s_mean = NA_real_,
        runtime_s_median = NA_real_
      ) %>%
      select(
        method,
        all_of(param_cols),
        runtime_s_mean,
        runtime_s_median
      )

    est <- bind_rows(
      est,
      truth_row
    )
  }


  # ---------------------------------------------------------------------------
  # Final formatting
  # ---------------------------------------------------------------------------

  est %>%
    mutate(
      across(
        where(is.numeric),
        ~ round(.x, 4)
      )
    ) %>%
    arrange(
      factor(
        method,
        levels = c(
          "TRUTH",
          "foce",
          "focei",
          "saem",
          "vi",
          "free",
          "amortized"
        )
      )
    )
}


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------

option_list <- list(

  make_option(
    "--phase0-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--phase1-csv",
    type = "character",
    default = NULL,
    help = paste(
      "Dense/sparse grid.",
      "Nonlinear rows are excluded automatically."
    )
  ),

  make_option(
    "--nonlinear-csv",
    type = "character",
    default = NULL,
    help = paste(
      "Nonlinear MM tier.",
      "May be the same file as --phase1-csv."
    )
  ),

  make_option(
    "--theoph-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--warfarin-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--deltaofv-free-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--deltaofv-amortized-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--psis-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--baseline-csv",
    type = "character",
    default = NULL,
    help = "VI vs FOCEI vs SAEM comparison."
  ),

  make_option(
    "--out",
    type = "character",
    default = "publication/tables"
  )
)


args <- parse_args(
  OptionParser(
    option_list = option_list
  )
)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

cat(
  paste(
    rep("=", 72),
    collapse = ""
  ),
  "\n",
  sep = ""
)

cat(
  "MANUSCRIPT TABLES\n"
)

cat(
  paste(
    rep("=", 72),
    collapse = ""
  ),
  "\n",
  sep = ""
)


# -----------------------------------------------------------------------------
# Phase 0
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Phase 1 grid
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Nonlinear
# -----------------------------------------------------------------------------

nonlinear_path <- if (
  !is.null(args$nonlinear_csv) &&
  !is.na(args$nonlinear_csv) &&
  args$nonlinear_csv != ""
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

    "nonlinear" %in%
      unique(df_nonlinear$scenario)

  } else {

    FALSE
  }

  if (has_nonlinear) {

    save_table(
      table_nonlinear(df_nonlinear),
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


# -----------------------------------------------------------------------------
# Real data
# -----------------------------------------------------------------------------

realdata_tables <- list()


df <- try_load(
  args$theoph_csv,
  "Real data: Theophylline"
)

if (!is.null(df)) {

  realdata_tables$theophylline <- table_realdata(
    df,
    "theophylline"
  )
}


df <- try_load(
  args$warfarin_csv,
  "Real data: Warfarin"
)

if (!is.null(df)) {

  realdata_tables$warfarin <- table_realdata(
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


# -----------------------------------------------------------------------------
# dOFV calibration
# -----------------------------------------------------------------------------

calib_tables <- list()


df <- try_load(
  args$deltaofv_free_csv,
  "dOFV calibration: free posterior"
)

if (!is.null(df)) {

  calib_tables$free <- table_deltaofv_calibration(
    df,
    "free"
  )
}


df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV calibration: amortized posterior"
)

if (!is.null(df)) {

  calib_tables$amortized <- table_deltaofv_calibration(
    df,
    "amortized"
  )
}


if (length(calib_tables) > 0) {

  save_table(
    bind_rows(calib_tables),
    "table_deltaofv_calibration",
    args$out
  )
}


# -----------------------------------------------------------------------------
# PSIS
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Baseline comparison
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

full_out_dir <- resolve_path(
  args$out
)

cat(
  "\n",
  paste(
    rep("=", 72),
    collapse = ""
  ),
  "\n",
  sep = ""
)

cat(
  "Done. Tables written to ",
  full_out_dir,
  "/\n",
  sep = ""
)

cat(
  "Skipped tables mean the source was not supplied or was not found.\n"
)