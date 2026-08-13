#!/usr/bin/env Rscript

# publication/make_tables.R
#
# Manuscript tables from the project's result CSVs.
#
# This version:
#   - uses here::here() for explicit project-root-relative paths
#   - provides sensible default input paths
#   - still allows command-line overrides
#   - skips missing files cleanly
#   - prints resolved paths
#   - writes CSV + Markdown versions of each table
#
# Run directly:
#
#   Rscript publication/make_tables.R
#
# Or override selected paths:
#
#   Rscript publication/make_tables.R \
#     --phase0-csv outputs/phase0_results.csv \
#     --out publication/tables


# =========================================================================
# PACKAGES
# =========================================================================

library(dplyr)
library(tidyr)
library(readr)
library(here)


# =========================================================================
# PATH HELPERS
# =========================================================================

resolve_project_path <- function(path) {

  if (is.null(path) || !nzchar(path)) {
    return(NULL)
  }

  # Leave absolute paths unchanged.
  if (
    grepl("^/", path) ||
    grepl("^[A-Za-z]:[/\\\\]", path)
  ) {
    return(path)
  }

  # Otherwise interpret paths relative to the project root.
  here(path)
}


# =========================================================================
# COMMAND-LINE ARGUMENTS
# =========================================================================

parse_args <- function() {

  cli <- commandArgs(trailingOnly = TRUE)

  # Default project-root-relative locations.
  #
  # This allows the script to be run directly without supplying any flags.
  args <- list(
    phase0_csv = here(
      "outputs",
      "phase0_results.csv"
    ),

    phase1_csv = here(
      "outputs",
      "phase1_results.csv"
    ),

    nonlinear_csv = here(
      "outputs",
      "phase1_nonlinear_results.csv"
    ),

    theoph_csv = here(
      "outputs",
      "phase2_realdata_theoph.csv"
    ),

    warfarin_csv = here(
      "outputs",
      "phase2_realdata_warfarin.csv"
    ),

    deltaofv_free_csv = here(
      "outputs",
      "phase2_deltaofv_free.csv"
    ),

    deltaofv_amortized_csv = here(
      "outputs",
      "phase2_deltaofv_amortized.csv"
    ),

    psis_csv = here(
      "outputs",
      "phase2_psis_results.csv"
    ),

    baseline_csv = here(
      "outputs",
      "phase2_baseline_comparison.csv"
    ),

    out = here(
      "publication",
      "tables"
    )
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

  while (i <= length(cli)) {

    flag <- cli[i]

    if (!(flag %in% names(flag_map))) {

      warning(
        "Unknown argument: ",
        flag
      )

      i <- i + 1
      next
    }

    if (i == length(cli)) {

      stop(
        "Missing value after ",
        flag
      )
    }

    key <- flag_map[[flag]]

    args[[key]] <- resolve_project_path(
      cli[i + 1]
    )

    i <- i + 2
  }

  args
}


# =========================================================================
# UTILITIES
# =========================================================================

try_load <- function(path, label) {

  if (is.null(path)) {

    message(
      "[skip] ",
      label,
      ": no path given"
    )

    return(NULL)
  }

  if (!file.exists(path)) {

    message(
      "[skip] ",
      label,
      ": ",
      normalizePath(
        path,
        mustWork = FALSE
      ),
      " not found"
    )

    return(NULL)
  }

  df <- read_csv(
    path,
    show_col_types = FALSE
  )

  message(
    "[loaded] ",
    label,
    ": ",
    normalizePath(path),
    " (",
    nrow(df),
    " rows)"
  )

  df
}


# =========================================================================
# MARKDOWN TABLE WRITER
# =========================================================================

df_to_markdown <- function(df) {

  if (ncol(df) == 0) {
    return("")
  }

  format_cell <- function(x) {

    if (is.na(x)) {
      return("NA")
    }

    x <- as.character(x)

    # Escape pipes so values cannot break the Markdown table.
    gsub(
      "\\|",
      "\\\\|",
      x
    )
  }


  header <- paste0(
    "| ",
    paste(
      names(df),
      collapse = " | "
    ),
    " |"
  )


  separator <- paste0(
    "| ",
    paste(
      rep(
        "---",
        ncol(df)
      ),
      collapse = " | "
    ),
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
            format_cell,
            character(1)
          ),
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


# =========================================================================
# SAVE TABLE
# =========================================================================

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
    paste0(
      name,
      ".csv"
    )
  )


  md_path <- file.path(
    out_dir,
    paste0(
      name,
      ".md"
    )
  )


  write_csv(
    df,
    csv_path,
    na = ""
  )


  writeLines(
    df_to_markdown(df),
    md_path
  )


  cat("\n")

  cat(
    "=== ",
    name,
    " ===\n",
    sep = ""
  )


  print(
    df,
    n = Inf,
    width = Inf
  )


  message(
    "  -> ",
    normalizePath(
      csv_path,
      mustWork = FALSE
    )
  )

  message(
    "  -> ",
    normalizePath(
      md_path,
      mustWork = FALSE
    )
  )
}


# =========================================================================
# TABLE 1: PHASE 0
# =========================================================================

table_phase0 <- function(df) {

  df %>%
    filter(
      startsWith(
        param,
        "om_"
      )
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


# =========================================================================
# TABLE 2: PHASE 1 GRID
# =========================================================================

table_phase1_grid <- function(df) {

  df %>%
    filter(
      startsWith(
        param,
        "om_"
      ),

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


# =========================================================================
# TABLE 3: NONLINEAR
# =========================================================================

table_nonlinear <- function(df) {

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


# =========================================================================
# TABLE 4: REAL DATA
# =========================================================================

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


  if (length(param_cols) == 0) {

    warning(
      "No recognized parameter columns found for ",
      dataset_name
    )

    return(
      tibble()
    )
  }


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
          (
            estimate_lo -
              estimate_hi
          ) /
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


# =========================================================================
# TABLE 5: dOFV CALIBRATION
# =========================================================================

table_deltaofv_calibration <- function(
    df,
    label
) {

  dofv <- df$dofv

  dofv <- dofv[
    !is.na(dofv)
  ]


  if (length(dofv) == 0) {

    warning(
      "No non-missing dOFV values for ",
      label
    )

    return(
      tibble()
    )
  }


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


  # 50:50 mixture:
  #
  #   0.5 * point mass at zero
  # + 0.5 * chi-square(df = 1)
  #
  # A 5% test for the mixture corresponds to the 90th percentile
  # of chi-square(1).

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


# =========================================================================
# TABLE 6: PSIS / ESS
# =========================================================================

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


# =========================================================================
# TABLE 7: VI vs FOCEI vs SAEM
# =========================================================================

table_baseline_comparison <- function(df) {

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
      paste(
        "Baseline comparison file contains",
        "none of the expected parameter columns."
      )
    )
  }


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

      .groups = "drop"
    )


  # -----------------------------------------------------------------------
  # Runtime
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
  # Truth row
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
        all_of(
          names(est)
        )
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
  strrep(
    "=",
    72
  ),
  "\n",
  sep = ""
)

cat(
  "MANUSCRIPT TABLES\n"
)

cat(
  strrep(
    "=",
    72
  ),
  "\n\n",
  sep = ""
)


cat(
  "Project root: ",
  normalizePath(
    here(),
    mustWork = FALSE
  ),
  "\n",
  sep = ""
)


cat(
  "Working directory: ",
  normalizePath(
    getwd(),
    mustWork = FALSE
  ),
  "\n",
  sep = ""
)


cat(
  "Table output directory: ",
  normalizePath(
    args$out,
    mustWork = FALSE
  ),
  "\n\n",
  sep = ""
)


dir.create(
  args$out,
  recursive = TRUE,
  showWarnings = FALSE
)


# =========================================================================
# PHASE 0
# =========================================================================

df <- try_load(
  args$phase0_csv,
  "Phase 0 (go/no-go)"
)


if (!is.null(df)) {

  tab <- table_phase0(df)

  if (nrow(tab) > 0) {

    save_table(
      tab,
      "table_phase0_headline",
      args$out
    )
  }
}


# =========================================================================
# PHASE 1 GRID
# =========================================================================

df <- try_load(
  args$phase1_csv,
  "Phase 1 grid (Q2/Q4, linear)"
)


if (!is.null(df)) {

  tab <- table_phase1_grid(df)

  if (nrow(tab) > 0) {

    save_table(
      tab,
      "table_phase1_q2q4",
      args$out
    )
  }
}


# =========================================================================
# NONLINEAR
# =========================================================================

df <- try_load(
  args$nonlinear_csv,
  "Phase 1 nonlinear (Q3)"
)


if (!is.null(df)) {

  has_nonlinear <-
    !"scenario" %in% names(df) ||
    "nonlinear" %in% df$scenario


  if (has_nonlinear) {

    tab <- table_nonlinear(df)

    if (nrow(tab) > 0) {

      save_table(
        tab,
        "table_phase1_q3_nonlinear",
        args$out
      )
    }

  } else {

    message(
      "[skip] Q3 nonlinear table: ",
      "no 'nonlinear' scenario rows in the given file"
    )
  }
}


# =========================================================================
# REAL DATA
# =========================================================================

realdata_tables <- list()


df <- try_load(
  args$theoph_csv,
  "Real data: Theophylline"
)


if (!is.null(df)) {

  tab <- table_realdata(
    df,
    "theophylline"
  )

  if (nrow(tab) > 0) {

    realdata_tables[[
      length(realdata_tables) + 1
    ]] <- tab
  }
}


df <- try_load(
  args$warfarin_csv,
  "Real data: Warfarin"
)


if (!is.null(df)) {

  tab <- table_realdata(
    df,
    "warfarin"
  )

  if (nrow(tab) > 0) {

    realdata_tables[[
      length(realdata_tables) + 1
    ]] <- tab
  }
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


# =========================================================================
# dOFV CALIBRATION
# =========================================================================

calib_tables <- list()


df <- try_load(
  args$deltaofv_free_csv,
  "dOFV calibration: free posterior"
)


if (!is.null(df)) {

  tab <- table_deltaofv_calibration(
    df,
    "free"
  )

  if (nrow(tab) > 0) {

    calib_tables[[
      length(calib_tables) + 1
    ]] <- tab
  }
}


df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV calibration: amortized posterior"
)


if (!is.null(df)) {

  tab <- table_deltaofv_calibration(
    df,
    "amortized"
  )

  if (nrow(tab) > 0) {

    calib_tables[[
      length(calib_tables) + 1
    ]] <- tab
  }
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


# =========================================================================
# PSIS / ESS
# =========================================================================

df <- try_load(
  args$psis_csv,
  "PSIS/ESS diagnostic"
)


if (!is.null(df)) {

  if (
    all(
      c(
        "arm",
        "ess"
      ) %in% names(df)
    )
  ) {

    save_table(
      table_psis(df),
      "table_psis_ess",
      args$out
    )

  } else {

    message(
      "[skip] PSIS table: ",
      "required columns 'arm' and/or 'ess' are missing"
    )
  }
}


# =========================================================================
# VI vs FOCEI vs SAEM
# =========================================================================

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


# =========================================================================
# FINISH
# =========================================================================

cat("\n")

cat(
  strrep(
    "=",
    72
  ),
  "\n",
  sep = ""
)


cat(
  "Done.\n"
)


cat(
  "Tables written to:\n",
  normalizePath(
    args$out,
    mustWork = FALSE
  ),
  "\n",
  sep = ""
)