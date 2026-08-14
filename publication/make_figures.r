# publication/make_figures.R
#
# Generate manuscript figures from the project's result CSVs.
#
# Figures contain no manuscript captions or descriptive supertitles.
# Only axis labels, legends, panel labels, and reference lines are included.
#
# Missing input files are skipped rather than causing an error.
#
# OUTPUT:
#   publication/figures/
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
# Set any path to NULL if that analysis has not yet been run.

phase0_csv <- here("outputs", "phase0_results.csv")

phase1_csv <- here("outputs", "phase1_results.csv")

nonlinear_csv <- here("outputs", "phase1_nonlinear_results.csv")

theoph_csv <- here("outputs", "phase2_realdata_theoph.csv")

warfarin_csv <- here("outputs", "phase2_realdata_warfarin.csv")

deltaofv_free_csv <- here("outputs", "phase2_deltaofv_free.csv")

deltaofv_amortized_csv <- here("outputs", "phase2_deltaofv_amortized.csv")

psis_csv <- here("outputs", "phase2_psis_results.csv")


# Output directory

out_dir <- here("publication", "figures")


# -------------------------------------------------------------------------
# Plot theme
# -------------------------------------------------------------------------

theme_manuscript <- function() {

  theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        linewidth = 0.3
      ),
      strip.background = element_blank(),
      strip.text = element_text(
        face = "bold"
      ),
      legend.title = element_blank(),
      legend.position = "right"
    )
}


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

try_load <- function(path, label) {

  if (is.null(path) || is.na(path) || path == "") {

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
      path,
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
    path,
    " (",
    nrow(df),
    " rows)"
  )

  df
}


save_figure <- function(
  plot,
  name,
  width,
  height,
  out_dir = out_dir
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

  message("-> ", path)
}


# Standard error of the mean.
#
# Matches pandas:
#   .agg(["mean", "sem"])

sem <- function(x) {

  x <- x[!is.na(x)]

  if (length(x) <= 1) {
    return(NA_real_)
  }

  sd(x) / sqrt(length(x))
}


# -------------------------------------------------------------------------
# Figure: Phase 0
# -------------------------------------------------------------------------

fig_phase0 <- function(df, out_dir = out_dir) {

  om <- df %>%
    filter(
      str_starts(param, "om_")
    )

  summary_df <- om %>%
    group_by(
      posterior,
      param,
      K
    ) %>%
    summarise(
      mean_bias = mean(rel_bias_pct, na.rm = TRUE),
      sem_bias = sem(rel_bias_pct),
      .groups = "drop"
    )

  n_posteriors <- n_distinct(
    summary_df$posterior
  )

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean_bias,
      group = param,
      color = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_errorbar(
      aes(
        ymin = mean_bias - sem_bias,
        ymax = mean_bias + sem_bias
      ),
      width = 0.08
    ) +
    geom_line(
      linewidth = 0.6
    ) +
    geom_point(
      size = 2
    ) +
    scale_x_log10(
      breaks = sort(unique(summary_df$K))
    ) +
    facet_wrap(
      vars(posterior),
      nrow = 1
    ) +
    labs(
      x = "K",
      y = "relative bias in omega (%)"
    ) +
    theme_manuscript()

  save_figure(
    p,
    "fig_phase0_omega_bias",
    width = 5.5 * n_posteriors,
    height = 4.2,
    out_dir = out_dir
  )
}


# -------------------------------------------------------------------------
# Figure: Phase 1 grid
# -------------------------------------------------------------------------
#
# IMPORTANT:
#
# This intentionally reproduces the Python behavior.
#
# The Python implementation filters to omega parameters but then computes:
#
#     groupby("K").rel_bias_pct.mean()
#
# within each posterior.
#
# Therefore om_CL, om_V, and om_ka are pooled together.
#
# If parameter-specific lines are desired instead, param should be added
# to the grouping below.

fig_phase1_grid <- function(df, out_dir = out_dir) {

  om <- df %>%
    filter(
      str_starts(param, "om_"),
      scenario %in% c(
        "dense",
        "sparse"
      )
    )

  summary_df <- om %>%
    group_by(
      scenario,
      family,
      posterior,
      K
    ) %>%
    summarise(
      mean_bias = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sem_bias = sem(
        rel_bias_pct
      ),
      .groups = "drop"
    )

  n_families <- n_distinct(
    summary_df$family
  )

  n_scenarios <- n_distinct(
    summary_df$scenario
  )

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean_bias,
      group = posterior,
      color = posterior
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_errorbar(
      aes(
        ymin = mean_bias - sem_bias,
        ymax = mean_bias + sem_bias
      ),
      width = 0.08
    ) +
    geom_line(
      linewidth = 0.6
    ) +
    geom_point(
      size = 2
    ) +
    scale_x_log10(
      breaks = sort(
        unique(summary_df$K)
      )
    ) +
    facet_grid(
      rows = vars(scenario),
      cols = vars(family),
      scales = "fixed"
    ) +
    labs(
      x = "K",
      y = "omega bias (%)"
    ) +
    theme_manuscript()

  save_figure(
    p,
    "fig_phase1_grid",
    width = 5.5 * n_families,
    height = 4 * n_scenarios,
    out_dir = out_dir
  )
}


# -------------------------------------------------------------------------
# Figure: nonlinear tier
# -------------------------------------------------------------------------

fig_nonlinear <- function(df, out_dir = out_dir) {

  if ("scenario" %in% names(df)) {

    df <- df %>%
      filter(
        scenario == "nonlinear"
      )
  }

  om <- df %>%
    filter(
      str_starts(param, "om_")
    )

  summary_df <- om %>%
    group_by(
      param,
      K
    ) %>%
    summarise(
      mean_bias = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sem_bias = sem(
        rel_bias_pct
      ),
      .groups = "drop"
    )

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean_bias,
      group = param,
      color = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_errorbar(
      aes(
        ymin = mean_bias - sem_bias,
        ymax = mean_bias + sem_bias
      ),
      width = 0.08
    ) +
    geom_line(
      linewidth = 0.6
    ) +
    geom_point(
      size = 2
    ) +
    scale_x_log10(
      breaks = sort(
        unique(summary_df$K)
      )
    ) +
    labs(
      x = "K",
      y = "relative bias in omega (%)"
    ) +
    theme_manuscript()

  save_figure(
    p,
    "fig_phase1_nonlinear",
    width = 6,
    height = 4.5,
    out_dir = out_dir
  )
}


# -------------------------------------------------------------------------
# Figure: real data
# -------------------------------------------------------------------------

prepare_realdata <- function(
  df,
  dataset_name
) {

  om_cols <- names(df)[
    str_starts(
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

  lo <- df %>%
    filter(
      K == k_lo
    ) %>%
    slice(1) %>%
    select(
      all_of(om_cols)
    ) %>%
    pivot_longer(
      everything(),
      names_to = "param",
      values_to = "estimate"
    ) %>%
    mutate(
      K = paste0(
        "K=",
        k_lo
      )
    )

  hi <- df %>%
    filter(
      K == k_hi
    ) %>%
    slice(1) %>%
    select(
      all_of(om_cols)
    ) %>%
    pivot_longer(
      everything(),
      names_to = "param",
      values_to = "estimate"
    ) %>%
    mutate(
      K = paste0(
        "K=",
        k_hi
      )
    )

  bind_rows(
    lo,
    hi
  ) %>%
    mutate(
      dataset = dataset_name,
      param = factor(
        param,
        levels = om_cols
      ),
      K = factor(
        K,
        levels = c(
          paste0("K=", k_lo),
          paste0("K=", k_hi)
        )
      )
    )
}


fig_realdata <- function(
  datasets,
  out_dir = out_dir
) {

  plot_df <- map_dfr(
    datasets,
    ~ prepare_realdata(
      .x$df,
      .x$name
    )
  )

  n_datasets <- n_distinct(
    plot_df$dataset
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
      vars(dataset),
      nrow = 1,
      scales = "free_x"
    ) +
    labs(
      x = NULL,
      y = "omega estimate"
    ) +
    theme_manuscript() +
    theme(
      panel.grid.major.x = element_blank()
    )

  save_figure(
    p,
    "fig_realdata_k1_vs_khigh",
    width = 5 * n_datasets,
    height = 4.2,
    out_dir = out_dir
  )
}


# -------------------------------------------------------------------------
# Figure: dOFV calibration
# -------------------------------------------------------------------------

fig_deltaofv <- function(
  conditions,
  out_dir = out_dir
) {

  empirical_df <- map_dfr(
    conditions,
    function(x) {

      x$df %>%
        transmute(
          condition = x$name,
          dofv = dofv
        )
    }
  )


  # Reference density:
  #
  #   0.5 * chi-square(df = 1)
  #
  # matching:
  #
  #   0.5 * stats.chi2.pdf(xx, df=1)

  max_dofv <- max(
    empirical_df$dofv,
    na.rm = TRUE
  )

  x_max <- max(
    max_dofv,
    8
  )

  reference_df <- crossing(
    condition = unique(
      empirical_df$condition
    ),
    dofv = seq(
      0.01,
      x_max,
      length.out = 300
    )
  ) %>%
    mutate(
      density = 0.5 * dchisq(
        dofv,
        df = 1
      )
    )

  n_conditions <- n_distinct(
    empirical_df$condition
  )

  p <- ggplot(
    empirical_df,
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
      data = reference_df,
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
      alpha = 0.5
    ) +
    facet_wrap(
      vars(condition),
      nrow = 1
    ) +
    labs(
      x = "dOFV",
      y = "density"
    ) +
    theme_manuscript()

  save_figure(
    p,
    "fig_deltaofv_calibration",
    width = 6 * n_conditions,
    height = 4.5,
    out_dir = out_dir
  )
}


# -------------------------------------------------------------------------
# Figure: PSIS ESS
# -------------------------------------------------------------------------

fig_psis <- function(df, out_dir = out_dir) {

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
      y = "count"
    ) +
    theme_manuscript()

  save_figure(
    p,
    "fig_psis_ess",
    width = 6,
    height = 4.5,
    out_dir = out_dir
  )
}


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

cat(
  "\n",
  str_dup("=", 72),
  "\nMANUSCRIPT FIGURES\n",
  str_dup("=", 72),
  "\n\n",
  sep = ""
)


# -------------------------------------------------------------------------
# Phase 0
# -------------------------------------------------------------------------

df_phase0 <- try_load(
  phase0_csv,
  "Phase 0"
)

if (!is.null(df_phase0)) {

  fig_phase0(
    df_phase0,
    out_dir
  )
}


# -------------------------------------------------------------------------
# Phase 1 grid
# -------------------------------------------------------------------------

df_phase1 <- try_load(
  phase1_csv,
  "Phase 1 grid"
)

if (!is.null(df_phase1)) {

  fig_phase1_grid(
    df_phase1,
    out_dir
  )
}


# -------------------------------------------------------------------------
# Nonlinear
#
# Use nonlinear_csv when available, otherwise fall back to phase1_csv.
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
  "Phase 1 nonlinear"
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

    fig_nonlinear(
      df_nonlinear,
      out_dir
    )

  } else {

    message(
      "[skip] nonlinear figure: ",
      "no 'nonlinear' scenario rows in the given file"
    )
  }
}


# -------------------------------------------------------------------------
# Real data
# -------------------------------------------------------------------------

realdata <- list()


df_theoph <- try_load(
  theoph_csv,
  "Theophylline"
)

if (!is.null(df_theoph)) {

  realdata <- append(
    realdata,
    list(
      list(
        name = "theophylline",
        df = df_theoph
      )
    )
  )
}


df_warfarin <- try_load(
  warfarin_csv,
  "warfarin"
)

if (!is.null(df_warfarin)) {

  realdata <- append(
    realdata,
    list(
      list(
        name = "warfarin",
        df = df_warfarin
      )
    )
  )
}


if (length(realdata) > 0) {

  fig_realdata(
    realdata,
    out_dir
  )
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

calib <- list()


df_deltaofv_free <- try_load(
  deltaofv_free_csv,
  "dOFV free"
)

if (!is.null(df_deltaofv_free)) {

  calib <- append(
    calib,
    list(
      list(
        name = "free",
        df = df_deltaofv_free
      )
    )
  )
}


df_deltaofv_amortized <- try_load(
  deltaofv_amortized_csv,
  "dOFV amortized"
)

if (!is.null(df_deltaofv_amortized)) {

  calib <- append(
    calib,
    list(
      list(
        name = "amortized",
        df = df_deltaofv_amortized
      )
    )
  )
}


if (length(calib) > 0) {

  fig_deltaofv(
    calib,
    out_dir
  )
}


# -------------------------------------------------------------------------
# PSIS
# -------------------------------------------------------------------------

df_psis <- try_load(
  psis_csv,
  "PSIS"
)

if (!is.null(df_psis)) {

  fig_psis(
    df_psis,
    out_dir
  )
}


# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

cat(
  "\n",
  str_dup("=", 72),
  "\n",
  "Done. Figures written to:\n",
  out_dir,
  "/\n",
  sep = ""
)