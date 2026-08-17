#!/usr/bin/env Rscript

# =============================================================================
# publication/make_tables.R
#
# Manuscript tables from already-generated project result CSVs.
#
# PURE POST-PROCESSING ONLY:
#   - reads result CSVs
#   - calculates manuscript summaries
#   - writes .csv and .md tables
#
# This script does NOT:
#   - run models
#   - run simulations
#   - call nlmixr2
#   - call Python
#   - source analysis scripts
#   - modify source result CSVs
#
# WHY EXPLICIT FILE PATHS:
#   Several analysis scripts may use the same default output filename for
#   different conditions. This script therefore never guesses which result
#   file represents which condition. Each condition is supplied explicitly.
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
# Relative paths are resolved from the project root using here::here().
#
# OUTPUT:
#   publication/tables/*.csv
#   publication/tables/*.md
#
# TABLE DESIGN:
#   - Phase 0: one combined table for structural fixed effects, residual
#     variability, and BSV SDs; K is shown across columns.
#   - Phase 1: same compact combined layout, additionally stratified by
#     sampling design, posterior architecture, and variational family.
#   - Nonlinear: one row per parameter; each K cell shows estimate (bias %).
#   - Real data: compact K=1 vs K=64 comparison.
#   - Baseline comparison: pharmacometrics-style orientation with parameters
#     as rows and methods as columns; median runtime shown as final row.
# =============================================================================


suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(here)
})


# =============================================================================
# Arguments
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


args <- parse_args(
  commandArgs(trailingOnly = TRUE)
)


# =============================================================================
# Path handling
# =============================================================================

is_absolute_path <- function(path) {

  if (is.null(path) || !nzchar(path)) {
    return(FALSE)
  }

  grepl(
    "^(/|~[/\\\\]|[A-Za-z]:[/\\\\])",
    path
  )
}


resolve_path <- function(path) {

  if (is.null(path) || !nzchar(path)) {
    return(NULL)
  }

  path <- path.expand(path)

  if (is_absolute_path(path)) {
    return(path)
  }

  here::here(path)
}


# =============================================================================
# Utilities
# =============================================================================

try_load <- function(
  path,
  label
) {

  if (is.null(path) || !nzchar(path)) {

    cat(
      sprintf(
        "\n[skip] %s: no path given\n",
        label
      )
    )

    return(NULL)
  }

  resolved <- resolve_path(path)

  if (!file.exists(resolved)) {

    cat(
      sprintf(
        "\n[skip] %s: %s not found\n",
        label,
        resolved
      )
    )

    return(NULL)
  }

  df <- read_csv(
    resolved,
    show_col_types = FALSE
  )

  cat(
    sprintf(
      "\n[loaded] %s: %s (%d rows)\n",
      label,
      resolved,
      nrow(df)
    )
  )

  df
}


# Approximately 3 significant digits, without padded trailing zeroes.
# Examples:
#   -1.160000 -> "-1.16"
#   -19.30000 -> "-19.3"
#   -83.00000 -> "-83"
#    0.039978 -> "0.04"
format_sig <- function(
  x,
  digits = 3
) {

  if (
    length(x) == 0 ||
    is.na(x) ||
    !is.finite(x)
  ) {
    return("")
  }

  value <- signif(
    as.numeric(x),
    digits = digits
  )

  if (value == 0) {
    return("0")
  }

  out <- format(
    value,
    scientific = FALSE,
    trim = TRUE,
    nsmall = 0
  )

  if (grepl("\\.", out)) {
    out <- sub("0+$", "", out)
    out <- sub("\\.$", "", out)
  }

  out
}


format_md_value <- function(x) {

  if (length(x) == 0 || is.na(x)) {
    return("")
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
    paste(
      rep("---", length(cols)),
      collapse = " | "
    ),
    " |"
  )

  rows <- vapply(
    seq_len(nrow(df)),
    function(i) {

      values <- vapply(
        df[i, , drop = FALSE],
        format_md_value,
        FUN.VALUE = character(1)
      )

      paste0(
        "| ",
        paste(
          values,
          collapse = " | "
        ),
        " |"
      )
    },
    FUN.VALUE = character(1)
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


save_table <- function(
  df,
  name,
  out_dir
) {

  df <- format_table_headers(df)

  out_dir <- resolve_path(out_dir)

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
# Publication notation
# =============================================================================

format_param_label <- function(x) {

  labels <- c(
    "CL" = "$CL$",
    "V" = "$V$",
    "ka" = "$k_a$",
    "sigma" = "$\\sigma$",
    "Vmax" = "$V_{\\max}$",
    "Km" = "$K_m$",
    "om_CL" = "$\\omega_{CL}$",
    "om_V" = "$\\omega_V$",
    "om_ka" = "$\\omega_{k_a}$",
    "om_Vmax" = "$\\omega_{V_{\\max}}$",
    "om_Km" = "$\\omega_{K_m}$"
  )

  out <- unname(
    labels[
      as.character(x)
    ]
  )

  missing <- is.na(out)

  out[missing] <-
    as.character(x)[missing]

  out
}


format_param_column <- function(df) {

  if ("param" %in% names(df)) {

    df <- df %>%
      mutate(
        param = format_param_label(
          param
        )
      )
  }

  df
}


format_table_headers <- function(df) {

  header_map <- c(
    "posterior" = "Posterior",
    "scenario" = "Sampling design",
    "family" = "Variational family",
    "parameter_type" = "Parameter type",
    "param" = "Parameter",
    "truth" = "Truth",
    "dataset" = "Dataset",
    "condition" = "Condition",
    "n_reps" = "Replicates",
    "boundary_fraction_pct" = "Boundary fraction (%)",
    "n_negative_dofv" = "Negative $\\Delta$OFV",
    "ks_stat" = "KS statistic",
    "ks_p" = "KS p-value",
    "type1_error_correct_test_pct" = "Type-I error (%)",
    "arm" = "Fit-quality arm",
    "mean" = "Mean",
    "median" = "Median",
    "min" = "Minimum",
    "max" = "Maximum",
    "method" = "Method"
  )

  new_names <- names(df)

  for (old_name in names(header_map)) {
    new_names[new_names == old_name] <-
      header_map[[old_name]]
  }

  names(df) <- new_names

  df
}


parameter_type <- function(param) {

  case_when(
    startsWith(
      as.character(param),
      "om_"
    ) ~ "BSV",
    as.character(param) == "sigma" ~ "RUV",
    TRUE ~ "Fixed effect"
  )
}


format_family <- function(x) {

  case_when(
    tolower(
      as.character(x)
    ) == "gaussian" ~ "Gaussian",
    tolower(
      as.character(x)
    ) == "flow" ~ "Flow",
    TRUE ~ as.character(x)
  )
}


# =============================================================================
# Table builders
# =============================================================================


# -----------------------------------------------------------------------------
# Phase 0
#
# One combined supplementary table:
#   Posterior | Parameter type | Parameter | K=1 | K=8 | K=64
#
# Each K cell is:
#   mean bias (SD)
#
# Replicate count is omitted because it is invariant within this experiment
# and is better stated once in the caption.
# -----------------------------------------------------------------------------

table_phase0_combined <- function(df) {

  df %>%
    mutate(
      parameter_type = parameter_type(
        param
      )
    ) %>%
    group_by(
      posterior,
      parameter_type,
      param,
      K
    ) %>%
    summarise(
      mean_bias = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sd_bias = sd(
        rel_bias_pct,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) %>%
    mutate(
      value = paste0(
        map_chr(
          mean_bias,
          format_sig
        ),
        " (",
        map_chr(
          sd_bias,
          format_sig
        ),
        ")"
      ),
      K_label = paste0(
        "$K=",
        K,
        "$"
      )
    ) %>%
    select(
      posterior,
      parameter_type,
      param,
      K_label,
      value
    ) %>%
    pivot_wider(
      names_from = K_label,
      values_from = value
    ) %>%
    mutate(
      parameter_type = factor(
        parameter_type,
        levels = c(
          "Fixed effect",
          "RUV",
          "BSV"
        )
      )
    ) %>%
    arrange(
      posterior,
      parameter_type,
      param
    ) %>%
    mutate(
      parameter_type = as.character(
        parameter_type
      )
    ) %>%
    format_param_column()
}


# -----------------------------------------------------------------------------
# Phase 1 linear grid
#
# One combined supplementary table:
#   Sampling design | Posterior | Family | Parameter type | Parameter
#                   | K=1 | K=8 | K=64
#
# Each K cell is:
#   mean bias (SD)
#
# Replicate count is omitted because it is invariant within this experiment.
# -----------------------------------------------------------------------------

table_phase1_by_design <- function(
  df,
  design
) {

  df %>%
    filter(
      scenario == design
    ) %>%
    mutate(
      parameter_type = parameter_type(
        param
      ),
      family = format_family(
        family
      )
    ) %>%
    group_by(
      posterior,
      family,
      parameter_type,
      param,
      K
    ) %>%
    summarise(
      mean_bias = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sd_bias = sd(
        rel_bias_pct,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) %>%
    mutate(
      value = paste0(
        map_chr(mean_bias, format_sig),
        " (",
        map_chr(sd_bias, format_sig),
        ")"
      ),
      K_label = paste0(
        "$K=",
        K,
        "$"
      )
    ) %>%
    select(
      posterior,
      family,
      parameter_type,
      param,
      K_label,
      value
    ) %>%
    pivot_wider(
      names_from = K_label,
      values_from = value
    ) %>%
    mutate(
      parameter_type = factor(
        parameter_type,
        levels = c(
          "Fixed effect",
          "RUV",
          "BSV"
        )
      )
    ) %>%
    arrange(
      posterior,
      family,
      parameter_type,
      param
    ) %>%
    mutate(
      parameter_type = as.character(
        parameter_type
      )
    ) %>%
    format_param_column()
}


# -----------------------------------------------------------------------------
# Nonlinear tier
#
# Compact table:
#   Parameter | Truth | K=1 | K=8 | K=64
#
# Each K cell is:
#   mean estimate (mean bias %)
#
# Posterior/family/n/fraction-converged columns are omitted because they are
# invariant in the production nonlinear analysis; state those facts once in
# the supplementary-table caption.
# -----------------------------------------------------------------------------

table_nonlinear <- function(df) {

  if ("scenario" %in% names(df)) {

    df <- df %>%
      filter(
        scenario == "nonlinear"
      )
  }

  df %>%
    group_by(
      param,
      K
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
      .groups = "drop"
    ) %>%
    mutate(
      truth_display = map_chr(
        truth,
        format_sig
      ),
      value = paste0(
        map_chr(
          mean_estimate,
          format_sig
        ),
        " (",
        map_chr(
          mean_bias_pct,
          format_sig
        ),
        "%)"
      ),
      K_label = paste0(
        "$K=",
        K,
        "$"
      )
    ) %>%
    select(
      param,
      truth_display,
      K_label,
      value
    ) %>%
    pivot_wider(
      names_from = K_label,
      values_from = value
    ) %>%
    rename(
      truth = truth_display
    ) %>%
    format_param_column()
}


# -----------------------------------------------------------------------------
# Real data
#
# Compact table:
#   Dataset | Parameter | K=1 | K=64 | Difference (%)
#
# No redundant K_low/K_high columns.
# Parameter estimates use approximately 3 significant digits.
# -----------------------------------------------------------------------------

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

  rows <- lapply(
    param_cols,
    function(p) {

      lo_value <- lo[[p]][[1]]
      hi_value <- hi[[p]][[1]]

      pct_diff <- if (
        !is.na(hi_value) &&
        is.finite(hi_value) &&
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
        K1 = format_sig(
          lo_value
        ),
        K64 = format_sig(
          hi_value
        ),
        difference_pct = format_sig(
          pct_diff
        )
      )
    }
  )

  out <- bind_rows(
    rows
  ) %>%
    format_param_column()

  names(out)[
    names(out) == "K1"
  ] <- paste0(
    "$K=",
    k_lo,
    "$"
  )

  names(out)[
    names(out) == "K64"
  ] <- paste0(
    "$K=",
    k_hi,
    "$"
  )

  names(out)[
    names(out) == "difference_pct"
  ] <- "Difference (%)"

  out
}


# -----------------------------------------------------------------------------
# dOFV calibration
# -----------------------------------------------------------------------------

table_deltaofv_calibration <- function(
  df,
  label
) {

  dofv <- df$dofv

  dofv <- dofv[
    is.finite(dofv)
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

  correct_cutoff <- qchisq(
    0.90,
    df = 1
  )

  type1_correct <- 100 *
    mean(
      dofv > correct_cutoff
    )

  tibble(
    condition = label,
    n_reps = length(
      dofv
    ),
    boundary_fraction_pct = format_sig(
      frac_boundary
    ),
    n_negative_dofv = sum(
      dofv < 0
    ),
    ks_stat = format_sig(
      ks_stat,
      digits = 4
    ),
    ks_p = if (
      is.na(ks_p)
    ) {
      ""
    } else if (
      ks_p < 0.0001
    ) {
      "<0.0001"
    } else {
      format_sig(
        ks_p,
        digits = 4
      )
    },
    type1_error_correct_test_pct = format_sig(
      type1_correct
    )
  )
}


# -----------------------------------------------------------------------------
# PSIS / ESS
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
        c(
          mean,
          median,
          min,
          max
        ),
        ~ map_chr(
          .x,
          format_sig
        )
      )
    )
}


# -----------------------------------------------------------------------------
# VI vs FOCEI vs SAEM
#
# PMx-style orientation:
#   Parameter | Truth | FOCEI | SAEM | VI, K=1 | VI, K=64
#
# Final row:
#   Median runtime (s)
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


  # Standardize runtime source.
  if ("runtime_s" %in% names(df)) {

    df <- df %>%
      mutate(
        runtime_s_std = runtime_s
      )

  } else if ("cpu_secs" %in% names(df)) {

    df <- df %>%
      mutate(
        runtime_s_std = cpu_secs
      )

  } else {

    df <- df %>%
      mutate(
        runtime_s_std = NA_real_
      )
  }


  # Publication-facing method labels.
  df <- df %>%
    mutate(
      method_display = case_when(
        method %in% c(
          "foce",
          "focei",
          "FOCE",
          "FOCEI"
        ) ~ "FOCEI",

        method %in% c(
          "saem",
          "SAEM"
        ) ~ "SAEM",

        method %in% c(
          "vi_K1",
          "vi_k1",
          "VI_K1",
          "vi1"
        ) ~ "VI, K = 1",

        method %in% c(
          "vi_K64",
          "vi_k64",
          "VI_K64",
          "vi64"
        ) ~ "VI, K = 64",

        TRUE ~ as.character(
          method
        )
      )
    )


  estimates <- df %>%
    group_by(
      method_display
    ) %>%
    summarise(
      across(
        all_of(
          param_cols
        ),
        ~ mean(
          .x,
          na.rm = TRUE
        )
      ),
      .groups = "drop"
    )


  estimates_long <- estimates %>%
    pivot_longer(
      cols = all_of(
        param_cols
      ),
      names_to = "param",
      values_to = "estimate"
    ) %>%
    pivot_wider(
      names_from = method_display,
      values_from = estimate
    )


  truth_values <- tibble(
    param = param_cols
  ) %>%
    mutate(
      Truth = map_dbl(
        param,
        function(p) {

          truth_name <- paste0(
            "truth_",
            p
          )

          if (
            truth_name %in% names(df)
          ) {

            value <- df[[truth_name]][[1]]

            if (
              length(value) == 1 &&
              is.finite(value)
            ) {
              return(
                as.numeric(
                  value
                )
              )
            }
          }

          NA_real_
        }
      )
    )


  out <- estimates_long %>%
    left_join(
      truth_values,
      by = "param"
    )


  desired_methods <- c(
    "Truth",
    "FOCEI",
    "SAEM",
    "VI, K = 1",
    "VI, K = 64"
  )

  for (nm in desired_methods) {

    if (
      !nm %in% names(out)
    ) {
      out[[nm]] <- NA_real_
    }
  }


  # Format values as compact strings rather than padded decimals.
  out <- out %>%
    mutate(
      Parameter = format_param_label(
        param
      )
    ) %>%
    select(
      Parameter,
      all_of(
        desired_methods
      )
    ) %>%
    mutate(
      across(
        all_of(
          desired_methods
        ),
        ~ map_chr(
          .x,
          format_sig
        )
      )
    )


  runtime <- df %>%
    filter(
      is.finite(
        runtime_s_std
      )
    ) %>%
    group_by(
      method_display
    ) %>%
    summarise(
      runtime = median(
        runtime_s_std,
        na.rm = TRUE
      ),
      .groups = "drop"
    )


  runtime_row <- tibble(
    Parameter = "Median runtime (s)",
    Truth = "",
    FOCEI = "",
    SAEM = "",
    `VI, K = 1` = "",
    `VI, K = 64` = ""
  )


  if (
    nrow(runtime) > 0
  ) {

    for (
      i in seq_len(
        nrow(runtime)
      )
    ) {

      method_name <-
        runtime$method_display[[i]]

      if (
        method_name %in%
          names(runtime_row)
      ) {

        runtime_row[[method_name]] <- format_sig(
          runtime$runtime[[i]]
        )
      }
    }
  }


  bind_rows(
    out,
    runtime_row
  )
}


# =============================================================================
# Main
# =============================================================================

cat(
  paste0(
    "\n",
    paste(
      rep(
        "=",
        72
      ),
      collapse = ""
    ),
    "\nMANUSCRIPT TABLES\n",
    paste(
      rep(
        "=",
        72
      ),
      collapse = ""
    ),
    "\n"
  )
)

cat(
  sprintf(
    "Project root: %s\n",
    here::here()
  )
)


# -----------------------------------------------------------------------------
# Phase 0 combined
# -----------------------------------------------------------------------------

df <- try_load(
  args$phase0_csv,
  "Phase 0"
)

if (!is.null(df)) {

  save_table(
    table_phase0_combined(
      df
    ),
    "table_phase0_combined",
    args$out
  )
}


# -----------------------------------------------------------------------------
# Phase 1 linear grid combined
# -----------------------------------------------------------------------------

df_phase1 <- try_load(
  args$phase1_csv,
  "Phase 1 grid"
)

if (!is.null(df_phase1)) {

  save_table(
    table_phase1_by_design(
      df_phase1,
      "dense"
    ),
    "table_phase1_dense",
    args$out
  )

  save_table(
    table_phase1_by_design(
      df_phase1,
      "sparse"
    ),
    "table_phase1_sparse",
    args$out
  )
}


# -----------------------------------------------------------------------------
# Phase 1 nonlinear
# -----------------------------------------------------------------------------

nonlinear_path <- if (
  !is.null(
    args$nonlinear_csv
  )
) {
  args$nonlinear_csv
} else {
  args$phase1_csv
}


df_nonlinear <- try_load(
  nonlinear_path,
  "Phase 1 nonlinear"
)


if (!is.null(df_nonlinear)) {

  has_nonlinear <- if (
    "scenario" %in%
      names(
        df_nonlinear
      )
  ) {

    any(
      df_nonlinear$scenario ==
        "nonlinear",
      na.rm = TRUE
    )

  } else {

    TRUE
  }


  if (has_nonlinear) {

    save_table(
      table_nonlinear(
        df_nonlinear
      ),
      "table_phase1_q3_nonlinear",
      args$out
    )

  } else {

    cat(
      "[skip] nonlinear table: ",
      "no 'nonlinear' scenario rows in the supplied file\n",
      sep = ""
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

  realdata_tables[[length(realdata_tables) + 1]] <- table_realdata(
    df,
    "theophylline"
  )
}


df <- try_load(
  args$warfarin_csv,
  "Real data: warfarin"
)

if (!is.null(df)) {

  realdata_tables[[length(realdata_tables) + 1]] <- table_realdata(
    df,
    "warfarin"
  )
}


if (
  length(
    realdata_tables
  ) > 0
) {

  save_table(
    bind_rows(
      realdata_tables
    ),
    "table_realdata_k1_vs_khigh",
    args$out
  )
}


# -----------------------------------------------------------------------------
# dOFV calibration
# -----------------------------------------------------------------------------

calibration_tables <- list()


df <- try_load(
  args$deltaofv_free_csv,
  "dOFV calibration: free posterior"
)

if (!is.null(df)) {

  calibration_tables[[length(calibration_tables) + 1]] <- table_deltaofv_calibration(
    df,
    "free"
  )
}


df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV calibration: amortized posterior"
)

if (!is.null(df)) {

  calibration_tables[[length(calibration_tables) + 1]] <- table_deltaofv_calibration(
    df,
    "amortized"
  )
}


if (
  length(
    calibration_tables
  ) > 0
) {

  save_table(
    bind_rows(
      calibration_tables
    ),
    "table_deltaofv_calibration",
    args$out
  )
}


# -----------------------------------------------------------------------------
# PSIS / ESS
# -----------------------------------------------------------------------------

df <- try_load(
  args$psis_csv,
  "PSIS/ESS diagnostic"
)

if (!is.null(df)) {

  save_table(
    table_psis(
      df
    ),
    "table_psis_ess",
    args$out
  )
}


# -----------------------------------------------------------------------------
# VI vs FOCEI vs SAEM
# -----------------------------------------------------------------------------

df <- try_load(
  args$baseline_csv,
  "VI vs FOCEI vs SAEM"
)

if (!is.null(df)) {

  save_table(
    table_baseline_comparison(
      df
    ),
    "table_baseline_comparison",
    args$out
  )
}


cat(
  paste0(
    "\n",
    paste(
      rep(
        "=",
        72
      ),
      collapse = ""
    ),
    "\nDone. Tables written to ",
    resolve_path(
      args$out
    ),
    "/\n",
    "Skipped tables mean the corresponding source was not supplied or found.\n"
  )
)
