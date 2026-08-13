#!/usr/bin/env Rscript

# publication/make_figures.R
#
# Manuscript figures from the project's result CSVs.
#
# Figures deliberately have NO captions or descriptive titles baked into
# the image. Only axis labels, legends, and panel labels required for
# interpretation are included.
#
# This version:
#   - uses here::here() for explicit project-root-relative paths
#   - provides sensible default input paths
#   - still allows command-line overrides
#   - skips missing files cleanly
#   - prints the resolved output directory
#   - saves both PNG and PDF versions of every figure
#
# Example:
#
# Rscript publication/make_figures.R
#
# or override selected paths:
#
# Rscript publication/make_figures.R \
#   --phase0-csv outputs/phase0_results.csv \
#   --out publication/figures


# =========================================================================
# PACKAGES
# =========================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(here)


# =========================================================================
# PUBLICATION THEME
# =========================================================================

theme_publication <- function(base_size = 11) {

  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),

      panel.grid.major = element_line(
        linewidth = 0.3,
        colour = "grey85"
      ),

      strip.background = element_blank(),

      strip.text = element_text(
        face = "bold"
      ),

      axis.title = element_text(
        size = base_size
      ),

      axis.text = element_text(
        size = base_size - 1
      ),

      legend.title = element_blank(),

      legend.position = "right"
    )
}


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

  # Interpret relative paths from the project root.
  here(path)
}


# =========================================================================
# COMMAND-LINE ARGUMENTS
# =========================================================================

parse_args <- function() {

  cli <- commandArgs(trailingOnly = TRUE)

  # Sensible defaults so the script can be run directly without supplying
  # command-line arguments.
  args <- list(
    phase0_csv = here("outputs", "phase0_results.csv"),
    phase1_csv = here("outputs", "phase1_results.csv"),
    nonlinear_csv = here("outputs", "phase1_nonlinear_results.csv"),
    theoph_csv = here("outputs", "phase2_realdata_theoph.csv"),
    warfarin_csv = here("outputs", "phase2_realdata_warfarin.csv"),
    deltaofv_free_csv = here("outputs", "phase2_deltaofv_free.csv"),
    deltaofv_amortized_csv = here("outputs", "phase2_deltaofv_amortized.csv"),
    psis_csv = here("outputs", "phase2_psis_results.csv"),
    out = here("publication", "figures")
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

  i <- 1

  while (i <= length(cli)) {

    flag <- cli[i]

    if (!(flag %in% names(flag_map))) {
      warning("Unknown argument: ", flag)
      i <- i + 1
      next
    }

    if (i == length(cli)) {
      stop("Missing value after ", flag)
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


savefig <- function(
    p,
    name,
    out_dir,
    width = 6,
    height = 4.5
) {

  dir.create(
    out_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  png_path <- file.path(
    out_dir,
    paste0(name, ".png")
  )

  pdf_path <- file.path(
    out_dir,
    paste0(name, ".pdf")
  )

  ggsave(
    filename = png_path,
    plot = p,
    width = width,
    height = height,
    units = "in",
    dpi = 300,
    bg = "white"
  )

  ggsave(
    filename = pdf_path,
    plot = p,
    width = width,
    height = height,
    units = "in",
    bg = "white"
  )

  message(
    "  -> ",
    normalizePath(
      png_path,
      mustWork = FALSE
    )
  )

  message(
    "  -> ",
    normalizePath(
      pdf_path,
      mustWork = FALSE
    )
  )
}


# =========================================================================
# FIGURE 1: PHASE 0
# =========================================================================

fig_phase0 <- function(df, out_dir) {

  om <- df %>%
    filter(
      grepl("^om_", param)
    )

  if (nrow(om) == 0) {

    message(
      "[skip] Phase 0 figure: no omega rows found"
    )

    return(invisible(NULL))
  }

  summary_df <- om %>%
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
          sum(
            !is.na(rel_bias_pct)
          )
        ),

      .groups = "drop"
    )

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean,
      colour = param,
      group = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_line(
      linewidth = 0.7
    ) +
    geom_point(
      size = 2.2
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.6
    ) +
    scale_x_log10(
      breaks = sort(
        unique(summary_df$K)
      ),
      labels = sort(
        unique(summary_df$K)
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
    theme_publication()

  n_post <- length(
    unique(summary_df$posterior)
  )

  savefig(
    p,
    "fig_phase0_omega_bias",
    out_dir,
    width = 5.5 * n_post,
    height = 4.2
  )
}


# =========================================================================
# FIGURE 2: PHASE 1 GRID
# =========================================================================

fig_phase1_grid <- function(df, out_dir) {

  om <- df %>%
    filter(
      grepl("^om_", param),
      scenario %in% c(
        "dense",
        "sparse"
      )
    )

  if (nrow(om) == 0) {

    message(
      "[skip] Phase 1 grid figure: no dense/sparse omega rows found"
    )

    return(invisible(NULL))
  }

  summary_df <- om %>%
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
          sum(
            !is.na(rel_bias_pct)
          )
        ),

      .groups = "drop"
    )

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean,
      colour = posterior,
      group = posterior
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_line(
      linewidth = 0.7
    ) +
    geom_point(
      size = 2.2
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.6
    ) +
    scale_x_log10(
      breaks = sort(
        unique(summary_df$K)
      ),
      labels = sort(
        unique(summary_df$K)
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
    theme_publication()

  n_scen <- length(
    unique(summary_df$scenario)
  )

  n_fam <- length(
    unique(summary_df$family)
  )

  savefig(
    p,
    "fig_phase1_grid",
    out_dir,
    width = 5.5 * n_fam,
    height = 4 * n_scen
  )
}


# =========================================================================
# FIGURE 3: NONLINEAR
# =========================================================================

fig_nonlinear <- function(df, out_dir) {

  nl <- df

  if ("scenario" %in% names(df)) {

    nl <- df %>%
      filter(
        scenario == "nonlinear"
      )
  }

  om <- nl %>%
    filter(
      grepl("^om_", param)
    )

  if (nrow(om) == 0) {

    message(
      "[skip] nonlinear figure: no omega rows found"
    )

    return(invisible(NULL))
  }

  summary_df <- om %>%
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
          sum(
            !is.na(rel_bias_pct)
          )
        ),

      .groups = "drop"
    )

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean,
      colour = param,
      group = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_line(
      linewidth = 0.7
    ) +
    geom_point(
      size = 2.2
    ) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.6
    ) +
    scale_x_log10(
      breaks = sort(
        unique(summary_df$K)
      ),
      labels = sort(
        unique(summary_df$K)
      )
    ) +
    labs(
      x = "K",
      y = "relative bias in omega (%)"
    ) +
    theme_publication()

  savefig(
    p,
    "fig_phase1_nonlinear",
    out_dir,
    width = 6,
    height = 4.5
  )
}


# =========================================================================
# FIGURE 4: REAL DATA
# =========================================================================

fig_realdata <- function(
    datasets,
    out_dir
) {

  plot_df <- bind_rows(
    lapply(
      datasets,
      function(x) {

        name <- x$name
        df <- x$data

        om_cols <- grep(
          "^om_",
          names(df),
          value = TRUE
        )

        if (length(om_cols) == 0) {
          return(NULL)
        }

        k_lo <- min(
          df$K,
          na.rm = TRUE
        )

        k_hi <- max(
          df$K,
          na.rm = TRUE
        )

        df %>%
          filter(
            K %in% c(
              k_lo,
              k_hi
            )
          ) %>%
          slice_head(
            n = 1,
            by = K
          ) %>%
          select(
            K,
            all_of(om_cols)
          ) %>%
          pivot_longer(
            cols = all_of(om_cols),
            names_to = "parameter",
            values_to = "estimate"
          ) %>%
          mutate(
            dataset = name,

            K_label = factor(
              paste0(
                "K=",
                K
              ),
              levels = c(
                paste0(
                  "K=",
                  k_lo
                ),
                paste0(
                  "K=",
                  k_hi
                )
              )
            )
          )
      }
    )
  )

  if (nrow(plot_df) == 0) {

    message(
      "[skip] real-data figure: no omega columns found"
    )

    return(invisible(NULL))
  }

  p <- ggplot(
    plot_df,
    aes(
      x = parameter,
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
      y = "omega estimate",
      fill = NULL
    ) +
    theme_publication()

  n <- length(
    unique(plot_df$dataset)
  )

  savefig(
    p,
    "fig_realdata_k1_vs_khigh",
    out_dir,
    width = 5 * n,
    height = 4.2
  )
}


# =========================================================================
# FIGURE 5: dOFV CALIBRATION
# =========================================================================

fig_deltaofv <- function(
    conditions,
    out_dir
) {

  plot_df <- bind_rows(
    lapply(
      conditions,
      function(x) {

        x$data %>%
          transmute(
            condition = x$name,
            dofv = dofv
          )
      }
    )
  )

  if (nrow(plot_df) == 0) {

    message(
      "[skip] dOFV figure: no observations found"
    )

    return(invisible(NULL))
  }

  xmax <- max(
    max(
      plot_df$dofv,
      na.rm = TRUE
    ),
    8
  )

  density_df <- bind_rows(
    lapply(
      unique(plot_df$condition),
      function(cond) {

        xx <- seq(
          0.01,
          xmax,
          length.out = 300
        )

        tibble(
          condition = cond,
          x = xx,
          density = 0.5 *
            dchisq(
              xx,
              df = 1
            )
        )
      }
    )
  )

  p <- ggplot(
    plot_df,
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
      data = density_df,
      aes(
        x = x,
        y = density,
        colour = "0.5 × chi-sq(1)"
      ),
      linewidth = 0.9,
      inherit.aes = FALSE
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    facet_wrap(
      ~ condition,
      nrow = 1
    ) +
    labs(
      x = "dOFV",
      y = "density",
      colour = NULL
    ) +
    theme_publication()

  n <- length(
    unique(plot_df$condition)
  )

  savefig(
    p,
    "fig_deltaofv_calibration",
    out_dir,
    width = 6 * n,
    height = 4.5
  )
}


# =========================================================================
# FIGURE 6: PSIS / ESS
# =========================================================================

fig_psis <- function(
    df,
    out_dir
) {

  if (
    !"ess" %in% names(df) ||
    !"arm" %in% names(df)
  ) {

    message(
      "[skip] PSIS figure: required columns 'ess' and/or 'arm' missing"
    )

    return(invisible(NULL))
  }

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
    theme_publication()

  savefig(
    p,
    "fig_psis_ess",
    out_dir,
    width = 6,
    height = 4.5
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
  "MANUSCRIPT FIGURES\n"
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
  "Figure output directory: ",
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


# -------------------------------------------------------------------------
# Phase 0
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Phase 1 grid
# -------------------------------------------------------------------------

df <- try_load(
  args$phase1_csv,
  "Phase 1 grid"
)

if (!is.null(df)) {

  fig_phase1_grid(
    df,
    args$out
  )
}


# -------------------------------------------------------------------------
# Nonlinear
# -------------------------------------------------------------------------

df <- try_load(
  args$nonlinear_csv,
  "Phase 1 nonlinear"
)

if (!is.null(df)) {

  has_nonlinear <-
    !"scenario" %in% names(df) ||
    "nonlinear" %in% df$scenario

  if (has_nonlinear) {

    fig_nonlinear(
      df,
      args$out
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


df <- try_load(
  args$theoph_csv,
  "Theophylline"
)

if (!is.null(df)) {

  realdata[[
    length(realdata) + 1
  ]] <- list(
    name = "theophylline",
    data = df
  )
}


df <- try_load(
  args$warfarin_csv,
  "Warfarin"
)

if (!is.null(df)) {

  realdata[[
    length(realdata) + 1
  ]] <- list(
    name = "warfarin",
    data = df
  )
}


if (length(realdata) > 0) {

  fig_realdata(
    realdata,
    args$out
  )
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

calib <- list()


df <- try_load(
  args$deltaofv_free_csv,
  "dOFV free"
)

if (!is.null(df)) {

  calib[[
    length(calib) + 1
  ]] <- list(
    name = "free",
    data = df
  )
}


df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV amortized"
)

if (!is.null(df)) {

  calib[[
    length(calib) + 1
  ]] <- list(
    name = "amortized",
    data = df
  )
}


if (length(calib) > 0) {

  fig_deltaofv(
    calib,
    args$out
  )
}


# -------------------------------------------------------------------------
# PSIS
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Finish
# -------------------------------------------------------------------------

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
  "Figures written to:\n",
  normalizePath(
    args$out,
    mustWork = FALSE
  ),
  "\n",
  sep = ""
)