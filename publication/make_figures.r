#!/usr/bin/env Rscript

# =============================================================================
# publication/make_figures.R
#
# Manuscript figures from already-generated project result CSVs.
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
#     2. summarizes values where required for plotting
#     3. creates manuscript figures
#     4. writes PNG files
#
# Figures intentionally contain NO manuscript captions or descriptive
# supertitles. Only axis labels, legends, and panel labels needed to
# interpret the figures are included.
#
# USAGE
#
# Rscript publication/make_figures.R \
#   --phase0-csv outputs/phase0_results.csv \
#   --phase1-csv outputs/phase1_results.csv \
#   --nonlinear-csv outputs/phase1_nonlinear_results.csv \
#   --theoph-csv outputs/phase2_realdata_theoph.csv \
#   --warfarin-csv outputs/phase2_realdata_warfarin.csv \
#   --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
#   --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized.csv \
#   --psis-csv outputs/phase2_psis_results.csv \
#   --out publication/figures
#
# Any input flag may be omitted. Its corresponding figure is skipped.
# =============================================================================


suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
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
    out = "publication/figures"
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
    cat(sprintf("[skip] %s: no path given\n", label))
    return(NULL)
  }

  if (!file.exists(path)) {
    cat(sprintf("[skip] %s: %s not found\n", label, path))
    return(NULL)
  }

  df <- read_csv(
    path,
    show_col_types = FALSE
  )

  cat(
    sprintf(
      "[loaded] %s: %s (%d rows)\n",
      label,
      path,
      nrow(df)
    )
  )

  df
}


savefig <- function(
  plot,
  name,
  out_dir,
  width,
  height
) {

  dir.create(
    out_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  path <- file.path(
    out_dir,
    paste0(name, ".png")
  )

  ggsave(
    filename = path,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    dpi = 300,
    bg = "white"
  )

  cat(
    sprintf(
      "-> %s\n",
      path
    )
  )
}


theme_manuscript <- function() {

  theme_bw(
    base_size = 11
  ) +
    theme(
      panel.grid.minor = element_blank(),
      legend.title = element_blank(),
      strip.background = element_rect(
        fill = "white"
      )
    )
}


mean_sem <- function(x) {

  n <- sum(
    !is.na(x)
  )

  tibble(
    mean = mean(
      x,
      na.rm = TRUE
    ),
    sem = if (
      n > 1
    ) {
      sd(
        x,
        na.rm = TRUE
      ) / sqrt(n)
    } else {
      NA_real_
    }
  )
}


# =============================================================================
# Figure builders
# =============================================================================

fig_phase0 <- function(
  df,
  out_dir
) {

  plot_df <- df %>%
    filter(
      startsWith(param, "om_")
    ) %>%
    group_by(
      posterior,
      param,
      K
    ) %>%
    summarise(
      mean = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sem = sd(
        rel_bias_pct,
        na.rm = TRUE
      ) /
        sqrt(
          sum(!is.na(rel_bias_pct))
        ),
      .groups = "drop"
    )

  p <- ggplot(
    plot_df,
    aes(
      x = K,
      y = mean,
      group = param,
      linetype = param,
      shape = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.4
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08
    ) +
    geom_line() +
    geom_point(
      size = 2
    ) +
    scale_x_log10(
      breaks = sort(
        unique(plot_df$K)
      )
    ) +
    facet_wrap(
      ~ posterior,
      nrow = 1
    ) +
    labs(
      x = "K",
      y = "relative bias in omega (%)"
    ) +
    theme_manuscript()

  n_post <- length(
    unique(plot_df$posterior)
  )

  savefig(
    p,
    "fig_phase0_omega_bias",
    out_dir,
    width = 5.5 * max(n_post, 1),
    height = 4.2
  )
}


fig_phase1_grid <- function(
  df,
  out_dir
) {

  plot_df <- df %>%
    filter(
      startsWith(param, "om_"),
      scenario %in% c(
        "dense",
        "sparse"
      )
    ) %>%
    group_by(
      scenario,
      family,
      posterior,
      K
    ) %>%
    summarise(
      mean = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sem = sd(
        rel_bias_pct,
        na.rm = TRUE
      ) /
        sqrt(
          sum(!is.na(rel_bias_pct))
        ),
      .groups = "drop"
    )

  p <- ggplot(
    plot_df,
    aes(
      x = K,
      y = mean,
      group = posterior,
      linetype = posterior,
      shape = posterior
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.4
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08
    ) +
    geom_line() +
    geom_point(
      size = 2
    ) +
    scale_x_log10(
      breaks = sort(
        unique(plot_df$K)
      )
    ) +
    facet_grid(
      rows = vars(scenario),
      cols = vars(family),
      scales = "free_y"
    ) +
    labs(
      x = "K",
      y = "omega bias (%)"
    ) +
    theme_manuscript()

  n_family <- length(
    unique(plot_df$family)
  )

  n_scenario <- length(
    unique(plot_df$scenario)
  )

  savefig(
    p,
    "fig_phase1_grid",
    out_dir,
    width = 5.5 * max(n_family, 1),
    height = 4 * max(n_scenario, 1)
  )
}


fig_nonlinear <- function(
  df,
  out_dir
) {

  if ("scenario" %in% names(df)) {

    df <- df %>%
      filter(
        scenario == "nonlinear"
      )
  }

  plot_df <- df %>%
    filter(
      startsWith(param, "om_")
    ) %>%
    group_by(
      param,
      K
    ) %>%
    summarise(
      mean = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sem = sd(
        rel_bias_pct,
        na.rm = TRUE
      ) /
        sqrt(
          sum(!is.na(rel_bias_pct))
        ),
      .groups = "drop"
    )

  p <- ggplot(
    plot_df,
    aes(
      x = K,
      y = mean,
      group = param,
      linetype = param,
      shape = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.4
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08
    ) +
    geom_line() +
    geom_point(
      size = 2
    ) +
    scale_x_log10(
      breaks = sort(
        unique(plot_df$K)
      )
    ) +
    labs(
      x = "K",
      y = "relative bias in omega (%)"
    ) +
    theme_manuscript()

  savefig(
    p,
    "fig_phase1_nonlinear",
    out_dir,
    width = 6,
    height = 4.5
  )
}


fig_realdata <- function(
  datasets,
  out_dir
) {

  plot_data <- list()

  for (name in names(datasets)) {

    df <- datasets[[name]]

    om_cols <- names(df)[
      startsWith(
        names(df),
        "om_"
      )
    ]

    k_lo <- min(
      df$K,
      na.rm = TRUE
    )

    k_hi <- max(
      df$K,
      na.rm = TRUE
    )

    sub <- df %>%
      filter(
        K %in% c(
          k_lo,
          k_hi
        )
      ) %>%
      select(
        K,
        all_of(om_cols)
      ) %>%
      slice_match(
        K,
        c(
          k_lo,
          k_hi
        )
      ) %>%
      pivot_longer(
        cols = all_of(om_cols),
        names_to = "param",
        values_to = "estimate"
      ) %>%
      mutate(
        dataset = name,
        K = paste0(
          "K=",
          K
        )
      )

    plot_data[[length(plot_data) + 1]] <- sub
  }

  plot_df <- bind_rows(
    plot_data
  )

  p <- ggplot(
    plot_df,
    aes(
      x = param,
      y = estimate,
      fill = K
    )
  ) +
    geom_col(
      position = position_dodge(
        width = 0.8
      ),
      width = 0.7
    ) +
    facet_wrap(
      ~ dataset,
      nrow = 1,
      scales = "free_x"
    ) +
    labs(
      x = NULL,
      y = "omega estimate",
      fill = NULL
    ) +
    theme_manuscript()

  n_dataset <- length(
    unique(plot_df$dataset)
  )

  savefig(
    p,
    "fig_realdata_k1_vs_khigh",
    out_dir,
    width = 5 * max(n_dataset, 1),
    height = 4.2
  )
}


fig_deltaofv <- function(
  conditions,
  out_dir
) {

  empirical <- bind_rows(
    lapply(
      names(conditions),
      function(label) {

        conditions[[label]] %>%
          transmute(
            condition = label,
            dofv = dofv
          )
      }
    )
  )

  xmax <- max(
    c(
      empirical$dofv,
      8
    ),
    na.rm = TRUE
  )

  reference <- expand_grid(
    condition = names(conditions),
    dofv = seq(
      0.01,
      xmax,
      length.out = 300
    )
  ) %>%
    mutate(
      density = 0.5 *
        dchisq(
          dofv,
          df = 1
        )
    )

  p <- ggplot(
    empirical,
    aes(
      x = dofv
    )
  ) +
    geom_histogram(
      aes(
        y = after_stat(density)
      ),
      bins = 30,
      alpha = 0.6
    ) +
    geom_line(
      data = reference,
      aes(
        x = dofv,
        y = density
      ),
      linewidth = 0.8,
      inherit.aes = FALSE
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.4
    ) +
    facet_wrap(
      ~ condition,
      nrow = 1
    ) +
    labs(
      x = "dOFV",
      y = "density"
    ) +
    theme_manuscript()

  n_condition <- length(
    unique(empirical$condition)
  )

  savefig(
    p,
    "fig_deltaofv_calibration",
    out_dir,
    width = 6 * max(n_condition, 1),
    height = 4.5
  )
}


fig_psis <- function(
  df,
  out_dir
) {

  p <- ggplot(
    df,
    aes(
      x = ess,
      fill = arm
    )
  ) +
    geom_histogram(
      bins = 20,
      alpha = 0.6,
      position = "identity"
    ) +
    labs(
      x = "effective sample size",
      y = "count",
      fill = NULL
    ) +
    theme_manuscript()

  savefig(
    p,
    "fig_psis_ess",
    out_dir,
    width = 6,
    height = 4.5
  )
}


# =============================================================================
# Main
# =============================================================================

cat(
  paste0(
    "\n",
    paste(rep("=", 72), collapse = ""),
    "\nMANUSCRIPT FIGURES\n",
    paste(rep("=", 72), collapse = ""),
    "\n"
  )
)


# Phase 0
df <- try_load(
  args$phase0_csv,
  "Phase 0"
)

if (!is.null(df)) {

  fig_phase0(
    df,
    args$out
  )
}


# Phase 1 grid
df_phase1 <- try_load(
  args$phase1_csv,
  "Phase 1 grid"
)

if (!is.null(df_phase1)) {

  fig_phase1_grid(
    df_phase1,
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
  "Phase 1 nonlinear"
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

    fig_nonlinear(
      df_nonlinear,
      args$out
    )

  } else {

    cat(
      "[skip] nonlinear figure: ",
      "no 'nonlinear' scenario rows in the given file\n",
      sep = ""
    )
  }
}


# Real data
realdata <- list()

df <- try_load(
  args$theoph_csv,
  "Theophylline"
)

if (!is.null(df)) {

  realdata[["theophylline"]] <- df
}

df <- try_load(
  args$warfarin_csv,
  "warfarin"
)

if (!is.null(df)) {

  realdata[["warfarin"]] <- df
}

if (length(realdata) > 0) {

  fig_realdata(
    realdata,
    args$out
  )
}


# dOFV calibration
calibration <- list()

df <- try_load(
  args$deltaofv_free_csv,
  "dOFV free"
)

if (!is.null(df)) {

  calibration[["free"]] <- df
}

df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV amortized"
)

if (!is.null(df)) {

  calibration[["amortized"]] <- df
}

if (length(calibration) > 0) {

  fig_deltaofv(
    calibration,
    args$out
  )
}


# PSIS
df <- try_load(
  args$psis_csv,
  "PSIS"
)

if (!is.null(df)) {

  fig_psis(
    df,
    args$out
  )
}


cat(
  sprintf(
    "\nDone. Figures written to %s/\n",
    args$out
  )
)