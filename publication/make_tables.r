# publication/make_tables.R
#
# Generate manuscript tables from the project's result CSVs.
#
# WHY EXPLICIT FILE PATHS:
# Several analysis scripts write to fixed default filenames regardless of
# condition. Running a second condition can therefore overwrite the first.
#
# This script never guesses which file corresponds to which condition.
# Each source file is specified explicitly below. If a file is not supplied
# or does not exist, the corresponding table is skipped.
#
# OUTPUT:
#   publication/tables/
#       table_phase0_headline.csv
#       table_phase0_headline.md
#       ...
#
# Dependencies:
#   tidyverse
#   here


# -------------------------------------------------------------------------
# Packages
# -------------------------------------------------------------------------

library(tidyverse)
library(here)


# -------------------------------------------------------------------------
# Input paths
# -------------------------------------------------------------------------
#
# Set a path to NULL if that analysis has not yet been run.
#
# here() makes all paths relative to the project root rather than the
# current working directory.

phase0_csv <- here("outputs", "phase0_results.csv")

phase1_csv <- here("outputs", "phase1_results.csv")

nonlinear_csv <- here("outputs", "phase1_nonlinear_results.csv")

theoph_csv <- here("outputs", "phase2_realdata_theoph.csv")

warfarin_csv <- here("outputs", "phase2_realdata_warfarin.csv")

deltaofv_free_csv <- here("outputs", "phase2_deltaofv_free.csv")

deltaofv_amortized_csv <- here("outputs", "phase2_deltaofv_amortized.csv")

psis_csv <- here("outputs", "phase2_psis_results.csv")

baseline_csv <- here("outputs", "phase2_baseline_comparison.csv")


# Output directory

out_dir <- here("publication", "tables")


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

try_load <- function(path, label) {

  if (is.null(path) || is.na(path) || path == "") {
    message("\n[skip] ", label, ": no path given")
    return(NULL)
  }

  if (!file.exists(path)) {
    message("\n[skip] ", label, ": ", path, " not found")
    return(NULL)
  }

  df <- read_csv(
    path,
    show_col_types = FALSE
  )

  message(
    "\n[loaded] ",
    label,
    ": ",
    path,
    " (",
    nrow(df),
    " rows)"
  )

  df
}


# Convert a data frame to a simple Markdown table.
#
# This avoids introducing another package dependency just for Markdown
# output.

df_to_markdown <- function(df) {

  df <- as.data.frame(df)

  format_value <- function(x) {

    if (is.na(x)) {
      return("NA")
    }

    as.character(x)
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
          map_chr(row, format_value),
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


save_table <- function(df, name, out_dir = out_dir) {

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
    na = "NA"
  )

  writeLines(
    df_to_markdown(df),
    md_path
  )

  cat("\n=== ", name, " ===\n", sep = "")

  print(
    df,
    n = Inf,
    width = Inf
  )

  cat(
    "\n-> ", csv_path,
    "\n-> ", md_path,
    "\n",
    sep = ""
  )
}


# -------------------------------------------------------------------------
# Table 1: Phase 0 omega bias
# -------------------------------------------------------------------------

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
      mean_bias_pct = mean(rel_bias_pct, na.rm = TRUE),
      sd_bias_pct = sd(rel_bias_pct, na.rm = TRUE),
      n_estimates = sum(!is.na(rel_bias_pct)),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(mean_bias_pct, sd_bias_pct),
        ~ round(.x, 2)
      )
    )
}


# -------------------------------------------------------------------------
# Phase 0 fixed effects companion table
# -------------------------------------------------------------------------

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
      mean_bias_pct = mean(rel_bias_pct, na.rm = TRUE),
      sd_bias_pct = sd(rel_bias_pct, na.rm = TRUE),
      n_estimates = sum(!is.na(rel_bias_pct)),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(mean_bias_pct, sd_bias_pct),
        ~ round(.x, 2)
      )
    )
}


# -------------------------------------------------------------------------
# Phase 1 Q2/Q4: omega bias
# -------------------------------------------------------------------------

table_phase1_grid <- function(df) {

  df %>%
    filter(
      str_starts(param, "om_"),
      scenario %in% c("dense", "sparse")
    ) %>%
    group_by(
      scenario,
      posterior,
      family,
      K,
      param
    ) %>%
    summarise(
      mean_bias_pct = mean(rel_bias_pct, na.rm = TRUE),
      sd_bias_pct = sd(rel_bias_pct, na.rm = TRUE),
      n_estimates = sum(!is.na(rel_bias_pct)),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(mean_bias_pct, sd_bias_pct),
        ~ round(.x, 2)
      )
    )
}


# -------------------------------------------------------------------------
# Phase 1 Q2/Q4 fixed effects
# -------------------------------------------------------------------------

table_phase1_grid_fixed_effects <- function(df) {

  df %>%
    filter(
      !str_starts(param, "om_"),
      scenario %in% c("dense", "sparse")
    ) %>%
    group_by(
      scenario,
      posterior,
      family,
      K,
      param
    ) %>%
    summarise(
      mean_bias_pct = mean(rel_bias_pct, na.rm = TRUE),
      sd_bias_pct = sd(rel_bias_pct, na.rm = TRUE),
      n_estimates = sum(!is.na(rel_bias_pct)),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(mean_bias_pct, sd_bias_pct),
        ~ round(.x, 2)
      )
    )
}


# -------------------------------------------------------------------------
# Phase 1 Q3: nonlinear model
# -------------------------------------------------------------------------

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
      mean_estimate = mean(estimate, na.rm = TRUE),
      truth = first(truth),
      mean_bias_pct = mean(rel_bias_pct, na.rm = TRUE),
      n = sum(!is.na(rel_bias_pct)),
      frac_converged = mean(converged, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        where(is.numeric),
        ~ round(.x, 3)
      )
    )
}


# -------------------------------------------------------------------------
# Real data: K = 1 vs K = high
# -------------------------------------------------------------------------

table_realdata <- function(df, dataset_name) {

  k_lo <- min(df$K, na.rm = TRUE)
  k_hi <- max(df$K, na.rm = TRUE)

  lo <- df %>%
    filter(K == k_lo) %>%
    slice(1)

  hi <- df %>%
    filter(K == k_hi) %>%
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

  map_dfr(
    param_cols,
    function(p) {

      lo_est <- lo[[p]][1]
      hi_est <- hi[[p]][1]

      pct_diff <- ifelse(
        is.na(hi_est) || hi_est == 0,
        NA_real_,
        100 * (lo_est - hi_est) / hi_est
      )

      tibble(
        dataset = dataset_name,
        param = p,
        K_low = k_lo,
        estimate_K_low = lo_est,
        K_high = k_hi,
        estimate_K_high = hi_est,
        pct_diff_low_vs_high = round(pct_diff, 2)
      )
    }
  )
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

table_deltaofv_calibration <- function(df, label) {

  dofv <- df$dofv

  frac_boundary <- 100 * mean(
    dofv <= 0.05,
    na.rm = TRUE
  )

  positive <- dofv[
    !is.na(dofv) &
      dofv > 0.05
  ]

  if (length(positive) >= 10) {

    ks_result <- ks.test(
      positive,
      "pchisq",
      df = 1
    )

    ks_stat <- unname(
      ks_result$statistic
    )

    ks_p <- ks_result$p.value

  } else {

    ks_stat <- NA_real_
    ks_p <- NA_real_
  }

  # 90th percentile of chi-square(1), corresponding to the
  # Self-Liang 50:50 mixture 95% test threshold.

  correct_cutoff <- qchisq(
    0.90,
    df = 1
  )

  type1_correct <- 100 * mean(
    dofv > correct_cutoff,
    na.rm = TRUE
  )

  n_negative <- sum(
    dofv < 0,
    na.rm = TRUE
  )

  tibble(
    condition = label,
    n_reps = sum(!is.na(dofv)),
    boundary_fraction_pct = round(frac_boundary, 1),
    n_negative_dofv = n_negative,
    ks_stat = round(ks_stat, 4),
    ks_p = round(ks_p, 4),
    type1_error_correct_test_pct = round(type1_correct, 1)
  )
}


# -------------------------------------------------------------------------
# PSIS / ESS
# -------------------------------------------------------------------------

table_psis <- function(df) {

  df %>%
    group_by(
      arm
    ) %>%
    summarise(
      mean = mean(ess, na.rm = TRUE),
      median = median(ess, na.rm = TRUE),
      min = min(ess, na.rm = TRUE),
      max = max(ess, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      across(
        c(mean, median, min, max),
        ~ round(.x, 1)
      )
    )
}


# -------------------------------------------------------------------------
# VI vs FOCEI vs SAEM
# -------------------------------------------------------------------------

table_baseline_comparison <- function(df) {

  candidate_cols <- c(
    "CL",
    "V",
    "ka",
    "om_CL",
    "om_V",
    "om_ka",
    "sigma"
  )

  param_cols <- intersect(
    candidate_cols,
    names(df)
  )

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

  truth_cols <- paste0(
    "truth_",
    param_cols
  )

  truth_cols <- intersect(
    truth_cols,
    names(df)
  )

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

    truth_row <- tibble(
      method = "TRUTH"
    ) %>%
      bind_cols(
        truth_values
      )

    # Make sure truth row has the same parameter columns.
    missing_cols <- setdiff(
      param_cols,
      names(truth_row)
    )

    for (col in missing_cols) {
      truth_row[[col]] <- NA_real_
    }

    truth_row <- truth_row %>%
      select(
        method,
        all_of(param_cols)
      )

    est <- bind_rows(
      est,
      truth_row
    )
  }

  if ("cpu_secs" %in% names(df)) {

    runtime <- df %>%
      group_by(
        method
      ) %>%
      summarise(
        cpu_secs_mean = mean(cpu_secs, na.rm = TRUE),
        cpu_secs_median = median(cpu_secs, na.rm = TRUE),
        .groups = "drop"
      )

    est <- est %>%
      left_join(
        runtime,
        by = "method"
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


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

cat(
  paste0(
    "\n",
    str_dup("=", 72),
    "\nMANUSCRIPT TABLES\n",
    str_dup("=", 72),
    "\n"
  )
)


# -------------------------------------------------------------------------
# Phase 0
# -------------------------------------------------------------------------

df_phase0 <- try_load(
  phase0_csv,
  "Phase 0 (go/no-go)"
)

if (!is.null(df_phase0)) {

  save_table(
    table_phase0(df_phase0),
    "table_phase0_headline"
  )

  save_table(
    table_phase0_fixed_effects(df_phase0),
    "table_phase0_fixed_effects"
  )
}


# -------------------------------------------------------------------------
# Phase 1 Q2/Q4
# -------------------------------------------------------------------------

df_phase1 <- try_load(
  phase1_csv,
  "Phase 1 grid (Q2/Q4, linear)"
)

if (!is.null(df_phase1)) {

  save_table(
    table_phase1_grid(df_phase1),
    "table_phase1_q2q4"
  )

  save_table(
    table_phase1_grid_fixed_effects(df_phase1),
    "table_phase1_q2q4_fixed_effects"
  )
}


# -------------------------------------------------------------------------
# Phase 1 Q3 nonlinear
#
# If nonlinear_csv does not exist, fall back to phase1_csv, matching the
# Python script's nonlinear_csv OR phase1_csv behavior.
# -------------------------------------------------------------------------

nonlinear_path <- if (
  !is.null(nonlinear_csv) &&
    file.exists(nonlinear_csv)
) {
  nonlinear_csv
} else {
  phase1_csv
}

df_nonlinear <- try_load(
  nonlinear_path,
  "Phase 1 nonlinear (Q3)"
)

if (!is.null(df_nonlinear)) {

  has_nonlinear <- (
    "scenario" %in% names(df_nonlinear) &&
      any(
        df_nonlinear$scenario == "nonlinear",
        na.rm = TRUE
      )
  )

  if (has_nonlinear) {

    save_table(
      table_nonlinear(df_nonlinear),
      "table_phase1_q3_nonlinear"
    )

  } else {

    message(
      "[skip] Q3 nonlinear table: ",
      "no 'nonlinear' scenario rows in the given file"
    )
  }
}


# -------------------------------------------------------------------------
# Real datasets
# -------------------------------------------------------------------------

realdata_tables <- list()


df_theoph <- try_load(
  theoph_csv,
  "Real data: Theophylline"
)

if (!is.null(df_theoph)) {

  realdata_tables <- append(
    realdata_tables,
    list(
      table_realdata(
        df_theoph,
        "theophylline"
      )
    )
  )
}


df_warfarin <- try_load(
  warfarin_csv,
  "Real data: warfarin"
)

if (!is.null(df_warfarin)) {

  realdata_tables <- append(
    realdata_tables,
    list(
      table_realdata(
        df_warfarin,
        "warfarin"
      )
    )
  )
}


if (length(realdata_tables) > 0) {

  save_table(
    bind_rows(realdata_tables),
    "table_realdata_k1_vs_khigh"
  )
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

calib_tables <- list()


df_deltaofv_free <- try_load(
  deltaofv_free_csv,
  "dOFV calibration: free posterior"
)

if (!is.null(df_deltaofv_free)) {

  calib_tables <- append(
    calib_tables,
    list(
      table_deltaofv_calibration(
        df_deltaofv_free,
        "free"
      )
    )
  )
}


df_deltaofv_amortized <- try_load(
  deltaofv_amortized_csv,
  "dOFV calibration: amortized posterior"
)

if (!is.null(df_deltaofv_amortized)) {

  calib_tables <- append(
    calib_tables,
    list(
      table_deltaofv_calibration(
        df_deltaofv_amortized,
        "amortized"
      )
    )
  )
}


if (length(calib_tables) > 0) {

  save_table(
    bind_rows(calib_tables),
    "table_deltaofv_calibration"
  )
}


# -------------------------------------------------------------------------
# PSIS / ESS
# -------------------------------------------------------------------------

df_psis <- try_load(
  psis_csv,
  "PSIS/ESS diagnostic"
)

if (!is.null(df_psis)) {

  save_table(
    table_psis(df_psis),
    "table_psis_ess"
  )
}


# -------------------------------------------------------------------------
# Baseline comparison
# -------------------------------------------------------------------------

df_baseline <- try_load(
  baseline_csv,
  "VI vs FOCEI vs SAEM"
)

if (!is.null(df_baseline)) {

  save_table(
    table_baseline_comparison(df_baseline),
    "table_baseline_comparison"
  )
}


# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

cat(
  "\n",
  str_dup("=", 72),
  "\n",
  "Done. Tables written to:\n",
  out_dir,
  "/\n\n",
  "Skipped tables simply mean the source was not supplied or was not found.\n",
  sep = ""
)