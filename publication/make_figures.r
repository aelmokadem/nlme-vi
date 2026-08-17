#!/usr/bin/env Rscript

# =============================================================================
# publication/make_figures.R
#
# Manuscript figures from already-generated project result CSVs.
#
# PURE POST-PROCESSING ONLY:
#   - reads result CSVs
#   - calculates summaries required for plotting
#   - creates manuscript PNG figures
#
# This script does NOT:
#   - run models
#   - run simulations
#   - call nlmixr2
#   - call Python
#   - source analysis scripts
#   - modify source result CSVs
#
# Figures deliberately contain NO manuscript captions or descriptive
# supertitles. Only axis labels, legends, and facet labels required to
# interpret each plot are included.
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
# Any flag may be omitted. Its corresponding figure is skipped.
#
# Relative paths are resolved from the project root using here::here().
# =============================================================================


suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
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
        "[skip] %s: no path given\n",
        label
      )
    )

    return(NULL)
  }

  resolved <- resolve_path(path)

  if (!file.exists(resolved)) {

    cat(
      sprintf(
        "[skip] %s: %s not found\n",
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
      "[loaded] %s: %s (%d rows)\n",
      label,
      resolved,
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

  out_dir <- resolve_path(out_dir)

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

  # Shared publication typography across all figures.
  # Explicit element sizes are used rather than relying only on base_size,
  # because multi-panel figures otherwise tend to look visually smaller
  # after scaling to manuscript width.

  theme_bw(
    base_size = 13
  ) +
    theme(
      axis.title = element_text(
        size = 13
      ),
      axis.text = element_text(
        size = 11
      ),
      legend.text = element_text(
        size = 11
      ),
      legend.title = element_blank(),
      legend.position = "top",
      strip.text = element_text(
        size = 12,
        face = "bold"
      ),
      strip.background = element_rect(
        fill = "white"
      ),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        linewidth = 0.3
      ),
      legend.key.height = grid::unit(
        0.7,
        "lines"
      ),
      legend.key.width = grid::unit(
        1.0,
        "lines"
      ),
      plot.margin = margin(
        8,
        10,
        8,
        8
      )
    )
}


sem_value <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (length(x) <= 1) {
    return(NA_real_)
  }

  sd(x) /
    sqrt(length(x))
}



# =============================================================================
# Publication notation
# =============================================================================

param_plotmath_labels <- function(values) {

  # Source CSVs use implementation-facing names such as om_CL.
  # Figures display the corresponding publication notation.  In particular,
  # om_* values are random-effect standard deviations (omega), whereas Omega
  # is reserved for the covariance matrix in the manuscript equations.

  label_text <- c(
    "CL" = "CL",
    "V" = "V",
    "ka" = "k[a]",
    "sigma" = "sigma",
    "Vmax" = "V[max]",
    "Km" = "K[m]",
    "om_CL" = "omega[CL]",
    "om_V" = "omega[V]",
    "om_ka" = "omega[k[a]]",
    "om_Vmax" = "omega[V[max]]",
    "om_Km" = "omega[K[m]]"
  )

  text <- unname(
    label_text[
      as.character(values)
    ]
  )

  missing <- is.na(text)
  text[missing] <- as.character(values)[missing]

  parse(
    text = text
  )
}


add_param_legend_scales <- function(plot) {

  plot +
    scale_color_discrete(
      labels = param_plotmath_labels
    ) +
    scale_shape_discrete(
      labels = param_plotmath_labels
    )
}


# =============================================================================
# Phase 0
# =============================================================================

fig_phase0 <- function(
  df,
  out_dir
) {

  plot_df <- df %>%
    filter(
      startsWith(
        as.character(param),
        "om_"
      )
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
      sem = sem_value(
        rel_bias_pct
      ),
      .groups = "drop"
    )

  p <- ggplot(
    plot_df,
      aes(
        x = K,
        y = mean,
        group = param,
        color = param,
        shape = param
      )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.55
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.55
    ) +
    geom_line(
      linewidth = 0.75
    ) +
    geom_point(
      size = 2.8
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
      y = expression("Relative bias in " * omega * " (%)")
    ) +
    scale_color_discrete(
      labels = param_plotmath_labels
    ) +
    scale_shape_discrete(
      labels = param_plotmath_labels
    ) +
    theme_manuscript()

  n_posteriors <- length(
    unique(plot_df$posterior)
  )

  savefig(
    p,
    "fig_phase0_omega_bias",
    out_dir,
    width = 10.5,
    height = 4.8
  )
}

# =============================================================================
# Phase 0 fixed-effect / residual-error validation
# =============================================================================

fig_phase0_fixed_effects <- function(
  df,
  out_dir
) {

  # Companion to the Phase 0 random-effect-SD figure.
  #
  # Shows the K-dependence (or lack thereof) of fixed effects and residual
  # error alongside the main random-effect-SD result. This is intentionally
  # descriptive: some parameters (e.g. ka) may show persistent K-independent
  # bias, while sigma can partially compensate for BSV at low K.
  #
  # Uses the exact same fits as fig_phase0(); only non-random-effect-SD parameters
  # are selected.

  plot_df <- df %>%
    filter(
      !startsWith(
        as.character(param),
        "om_"
      )
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
      sem = sem_value(
        rel_bias_pct
      ),
      .groups = "drop"
    )
  
  plot_df <- plot_df %>%
  mutate(
    param = factor(
      param,
      levels = c(
        "CL",
        "V",
        "ka",
        "sigma"
      )
    )
  )

  if (nrow(plot_df) == 0) {

    cat(
      "[skip] Phase 0 fixed-effect figure: no non-random-effect-SD rows found\n"
    )

    return(invisible(NULL))
  }


  p <- ggplot(
    plot_df,
    aes(
      x = K,
      y = mean,
      group = param,
      color = param,
      shape = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.55
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.55
    ) +
    geom_line(
      linewidth = 0.75
    ) +
    geom_point(
      size = 2.8
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
      y = "Relative bias (%)",
      color = NULL,
      shape = NULL
    ) +
    scale_color_discrete(
      labels = param_plotmath_labels
    ) +
    scale_shape_discrete(
      labels = param_plotmath_labels
    ) +
    theme_manuscript()


  n_posteriors <- length(
    unique(plot_df$posterior)
  )

  savefig(
    p,
    "fig_phase0_fixed_effects",
    out_dir,
    width = 10.5,
    height = 4.8
  )
}

# =============================================================================
# Phase 1 linear grid
# =============================================================================

fig_phase1_grid <- function(
  df,
  out_dir
) {

  # This intentionally mirrors the Python figure:
  #
  #   1. keep random-effect-SD rows
  #   2. keep dense/sparse scenarios
  #   3. within each scenario/family/posterior,
  #      average rel_bias_pct over replicates AND random-effect-SD parameters at each K
  #
  # The manuscript table retains parameter-specific results separately.

  om <- df %>%
    filter(
      startsWith(
        as.character(param),
        "om_"
      ),
      scenario %in% c(
        "dense",
        "sparse"
      )
    )

  if (nrow(om) == 0) {

    cat(
      "[skip] Phase 1 grid figure: no dense/sparse random-effect-SD rows found\n"
    )

    return(invisible(NULL))
  }

  # Helpful diagnostic: these are the raw combinations in the CSV BEFORE
  # summarization. This makes any difference from the Python result obvious.
  cat(
    "\nPhase 1 raw figure combinations:\n"
  )

  print(
    om %>%
      count(
        scenario,
        family,
        posterior,
        name = "n_rows"
      ) %>%
      arrange(
        scenario,
        family,
        posterior
      ),
    n = Inf
  )


  plot_df <- om %>%
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
      sem = sem_value(
        rel_bias_pct
      ),
      .groups = "drop"
    )


  cat(
    "\nPhase 1 plotted combinations:\n"
  )

  print(
    plot_df %>%
      count(
        scenario,
        family,
        posterior,
        name = "n_K_values"
      ) %>%
      arrange(
        scenario,
        family,
        posterior
      ),
    n = Inf
  )


  p <- ggplot(
    plot_df,
      aes(
        x = K,
        y = mean,
        group = posterior,
        color = posterior,
        shape = posterior
      )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.55
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.55
    ) +
    geom_line(
      linewidth = 0.75
    ) +
    geom_point(
      size = 2.8
    ) +
    scale_x_log10(
      breaks = sort(
        unique(plot_df$K)
      )
    ) +
    facet_grid(
      rows = vars(scenario),
      cols = vars(family)
    ) +
    labs(
      x = "K",
      y = expression("Relative bias in " * omega * " (%)")
    ) +
    theme_manuscript()


  n_families <- length(
    unique(plot_df$family)
  )

  n_scenarios <- length(
    unique(plot_df$scenario)
  )

  savefig(
    p,
    "fig_phase1_grid",
    out_dir,
    width = 10.5,
    height = 7.5
  )
}

# =============================================================================
# Phase 1 fixed-effect / residual-error validation
# =============================================================================

fig_phase1_grid_fixed_effects <- function(
  df,
  out_dir
) {

  # Companion to the Phase 1 random-effect-SD grid.
  #
  # Rows    = sampling scenario
  # Columns = variational family
  #
  # Individual lines represent fixed effects / residual error.
  #
  # Posterior architecture is shown using facets within each
  # scenario/family combination so that parameter-specific behavior
  # is not averaged away.

  fe <- df %>%
    filter(
      !startsWith(
        as.character(param),
        "om_"
      ),
      scenario %in% c(
        "dense",
        "sparse"
      )
    )

  if (nrow(fe) == 0) {

    cat(
      "[skip] Phase 1 fixed-effect figure: ",
      "no dense/sparse non-random-effect-SD rows found\n",
      sep = ""
    )

    return(invisible(NULL))
  }


  cat(
    "\nPhase 1 fixed-effect figure combinations:\n"
  )

  print(
    fe %>%
      count(
        scenario,
        family,
        posterior,
        param,
        name = "n_rows"
      ) %>%
      arrange(
        scenario,
        family,
        posterior,
        param
      ),
    n = Inf
  )


  plot_df <- fe %>%
    group_by(
      scenario,
      family,
      posterior,
      param,
      K
    ) %>%
    summarise(
      mean = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sem = sem_value(
        rel_bias_pct
      ),
      .groups = "drop"
    )
  
  plot_df <- plot_df %>%
  mutate(
    param = factor(
      param,
      levels = c(
        "CL",
        "V",
        "ka",
        "sigma"
      )
    )
  )

  p <- ggplot(
    plot_df,
    aes(
      x = K,
      y = mean,
      group = param,
      color = param,
      shape = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.55
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.55
    ) +
    geom_line(
      linewidth = 0.75
    ) +
    geom_point(
      size = 2.8
    ) +
    scale_x_log10(
      breaks = sort(
        unique(plot_df$K)
      )
    ) +
    facet_grid(
      rows = vars(scenario),
      cols = vars(family, posterior)
    ) +
    labs(
      x = "K",
      y = "Relative bias (%)",
      color = NULL,
      shape = NULL
    ) +
    scale_color_discrete(
      labels = param_plotmath_labels
    ) +
    scale_shape_discrete(
      labels = param_plotmath_labels
    ) +
    theme_manuscript()


  n_families <- length(
    unique(plot_df$family)
  )

  n_posteriors <- length(
    unique(plot_df$posterior)
  )

  n_scenarios <- length(
    unique(plot_df$scenario)
  )


  savefig(
    p,
    "fig_phase1_grid_fixed_effects",
    out_dir,
    width = 13.5,
    height = 7.5
  )
}

# =============================================================================
# Nonlinear tier
# =============================================================================

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

  om <- df %>%
    filter(
      startsWith(
        as.character(param),
        "om_"
      )
    )

  if (nrow(om) == 0) {

    cat(
      "[skip] nonlinear figure: no random-effect-SD rows found\n"
    )

    return(invisible(NULL))
  }


  plot_df <- om %>%
    group_by(
      param,
      K
    ) %>%
    summarise(
      mean = mean(
        rel_bias_pct,
        na.rm = TRUE
      ),
      sem = sem_value(
        rel_bias_pct
      ),
      .groups = "drop"
    )


  p <- ggplot(
    plot_df,
      aes(
        x = K,
        y = mean,
        group = param,
        color = param,
        shape = param
      )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.55
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.55
    ) +
    geom_line(
      linewidth = 0.75
    ) +
    geom_point(
      size = 2.8
    ) +
    scale_x_log10(
      breaks = sort(
        unique(plot_df$K)
      )
    ) +
    labs(
      x = "K",
      y = expression("Relative bias in " * omega * " (%)")
    ) +
    scale_color_discrete(
      labels = param_plotmath_labels
    ) +
    scale_shape_discrete(
      labels = param_plotmath_labels
    ) +
    theme_manuscript()


  savefig(
    p,
    "fig_phase1_nonlinear",
    out_dir,
    width = 7.0,
    height = 5.0
  )
}


# =============================================================================
# Real data
# =============================================================================

fig_realdata <- function(
  datasets,
  out_dir
) {

  plot_data <- list()


  for (dataset_name in names(datasets)) {

    df <- datasets[[dataset_name]]

    om_cols <- names(df)[
      startsWith(
        names(df),
        "om_"
      )
    ]

    if (length(om_cols) == 0) {
      next
    }

    k_lo <- min(
      df$K,
      na.rm = TRUE
    )

    k_hi <- max(
      df$K,
      na.rm = TRUE
    )


    # No slice_match() here.
    # This is deliberately simple and compatible with standard dplyr.
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
      pivot_longer(
        cols = all_of(om_cols),
        names_to = "param",
        values_to = "estimate"
      ) %>%
      mutate(
        dataset = dataset_name,
        K_label = paste0(
          "K=",
          K
        )
      )


    plot_data[[length(plot_data) + 1]] <-
      sub
  }


  if (length(plot_data) == 0) {

    cat(
      "[skip] real-data figure: no random-effect-SD columns found\n"
    )

    return(invisible(NULL))
  }


  plot_df <- bind_rows(
    plot_data
  )


  p <- ggplot(
    plot_df,
    aes(
      x = param,
      y = estimate,
      fill = K_label
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
      y = expression(omega ~ " estimate"),
      fill = NULL
    ) +
    scale_x_discrete(
      labels = param_plotmath_labels
    ) +
    theme_manuscript()


  n_datasets <- length(
    unique(plot_df$dataset)
  )

  savefig(
    p,
    "fig_realdata_k1_vs_khigh",
    out_dir,
    width = 10.0,
    height = 4.8
  )
}


# =============================================================================
# dOFV calibration
# =============================================================================

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


  empirical <- empirical %>%
    filter(
      is.finite(dofv)
    )


  if (nrow(empirical) == 0) {

    cat(
      "[skip] dOFV figure: no finite dOFV values found\n"
    )

    return(invisible(NULL))
  }


  xmax <- max(
    c(
      empirical$dofv,
      8
    ),
    na.rm = TRUE
  )


  reference <- expand_grid(
    condition = unique(
      empirical$condition
    ),
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
      linewidth = 0.9,
      inherit.aes = FALSE
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.55
    ) +
    facet_wrap(
      ~ condition,
      nrow = 1
    ) +
    labs(
      x = expression(Delta * "OFV"),
      y = "Density"
    ) +
    theme_manuscript()


  n_conditions <- length(
    unique(empirical$condition)
  )

  savefig(
    p,
    "fig_deltaofv_calibration",
    out_dir,
    width = ifelse(n_conditions > 1, 11.0, 7.0),
    height = 5.0
  )
}


# =============================================================================
# PSIS / ESS
# =============================================================================

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
      x = "Effective sample size",
      y = "Count",
      fill = NULL
    ) +
    theme_manuscript()


  savefig(
    p,
    "fig_psis_ess",
    out_dir,
    width = 7.0,
    height = 5.0
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

cat(
  sprintf(
    "Project root: %s\n",
    here::here()
  )
)


# -----------------------------------------------------------------------------
# Phase 0
# -----------------------------------------------------------------------------

df <- try_load(
  args$phase0_csv,
  "Phase 0"
)

if (!is.null(df)) {

  # Main random-effect-SD-bias figure
  fig_phase0(
    df,
    args$out
  )

  # Supplementary validation figure
  fig_phase0_fixed_effects(
    df,
    args$out
  )
}


# -----------------------------------------------------------------------------
# Phase 1 grid
# -----------------------------------------------------------------------------

df_phase1 <- try_load(
  args$phase1_csv,
  "Phase 1 grid"
)

if (!is.null(df_phase1)) {

  # Main random-effect-SD-bias figure
  fig_phase1_grid(
    df_phase1,
    args$out
  )

  # Supplementary validation figure
  fig_phase1_grid_fixed_effects(
    df_phase1,
    args$out
  )
}


# -----------------------------------------------------------------------------
# Nonlinear
# -----------------------------------------------------------------------------

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
      "no 'nonlinear' scenario rows in the supplied file\n",
      sep = ""
    )
  }
}


# -----------------------------------------------------------------------------
# Real data
# -----------------------------------------------------------------------------

realdata <- list()


df <- try_load(
  args$theoph_csv,
  "Theophylline"
)

if (!is.null(df)) {

  realdata[["theophylline"]] <-
    df
}


df <- try_load(
  args$warfarin_csv,
  "warfarin"
)

if (!is.null(df)) {

  realdata[["warfarin"]] <-
    df
}


if (length(realdata) > 0) {

  fig_realdata(
    realdata,
    args$out
  )
}


# -----------------------------------------------------------------------------
# dOFV calibration
# -----------------------------------------------------------------------------

calibration <- list()


df <- try_load(
  args$deltaofv_free_csv,
  "dOFV free"
)

if (!is.null(df)) {

  calibration[["free"]] <-
    df
}


df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV amortized"
)

if (!is.null(df)) {

  calibration[["amortized"]] <-
    df
}


if (length(calibration) > 0) {

  fig_deltaofv(
    calibration,
    args$out
  )
}


# -----------------------------------------------------------------------------
# PSIS
# -----------------------------------------------------------------------------

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
    resolve_path(args$out)
  )
)