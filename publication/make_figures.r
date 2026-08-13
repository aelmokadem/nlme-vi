#!/usr/bin/env Rscript

# publication/make_figures.R
#
# Manuscript figures from the project's result CSVs.
#
# Figures deliberately have NO captions or descriptive titles baked into
# the image. Only axis labels, legends, and panel labels required for
# interpretation are included.
#
# USAGE
#
# Rscript publication/make_figures.R \
#   --phase0-csv outputs/phase0_results.csv \
#   --phase1-csv outputs/phase1_results.csv \
#   --theoph-csv outputs/phase2_realdata_theoph.csv \
#   --warfarin-csv outputs/phase2_realdata_warfarin.csv \
#   --deltaofv-free-csv outputs/phase2_deltaofv_free.csv \
#   --deltaofv-amortized-csv outputs/phase2_deltaofv_amortized.csv \
#   --psis-csv outputs/phase2_psis_results.csv \
#   --out publication/figures
#
# Any flag can be omitted; that figure is skipped, not an error.


# -------------------------------------------------------------------------
# Packages
# -------------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)


# -------------------------------------------------------------------------
# Publication theme
# -------------------------------------------------------------------------

theme_publication <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        linewidth = 0.3,
        colour = "grey85"
      ),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 1),
      legend.title = element_blank(),
      legend.position = "right"
    )
}


# -------------------------------------------------------------------------
# Command-line argument handling
# -------------------------------------------------------------------------

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

  i <- 1

  while (i <= length(args)) {

    flag <- args[i]

    if (flag %in% names(flag_map)) {

      if (i == length(args)) {
        stop("Missing value after ", flag)
      }

      defaults[[flag_map[[flag]]]] <- args[i + 1]
      i <- i + 2

    } else {

      warning("Unknown argument: ", flag)
      i <- i + 1
    }
  }

  defaults
}


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

try_load <- function(path, label) {

  if (is.null(path)) {
    message("[skip] ", label, ": no path given")
    return(NULL)
  }

  if (!file.exists(path)) {
    message("[skip] ", label, ": ", path, " not found")
    return(NULL)
  }

  df <- read_csv(path, show_col_types = FALSE)

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

  path <- file.path(
    out_dir,
    paste0(name, ".png")
  )

  ggsave(
    filename = path,
    plot = p,
    width = width,
    height = height,
    units = "in",
    dpi = 300,
    bg = "white"
  )

  message("  -> ", path)
}


# =========================================================================
# FIGURES
# =========================================================================


# -------------------------------------------------------------------------
# Phase 0
# -------------------------------------------------------------------------

fig_phase0 <- function(df, out_dir) {

  # Omega bias vs K, one panel per posterior.

  om <- df %>%
    filter(grepl("^om_", param))

  summary_df <- om %>%
    group_by(posterior, param, K) %>%
    summarise(
      mean = mean(rel_bias_pct, na.rm = TRUE),
      sem = sd(rel_bias_pct, na.rm = TRUE) / sqrt(sum(!is.na(rel_bias_pct))),
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
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.2) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.6
    ) +
    scale_x_log10(
      breaks = sort(unique(summary_df$K)),
      labels = sort(unique(summary_df$K))
    ) +
    facet_wrap(
      ~posterior,
      nrow = 1
    ) +
    labs(
      x = "K",
      y = "relative bias in omega (%)"
    ) +
    theme_publication()

  n_post <- length(unique(summary_df$posterior))

  savefig(
    p,
    "fig_phase0_omega_bias",
    out_dir,
    width = 5.5 * n_post,
    height = 4.2
  )
}


# -------------------------------------------------------------------------
# Phase 1 grid
# -------------------------------------------------------------------------

fig_phase1_grid <- function(df, out_dir) {

  # Omega bias vs K:
  # scenario (rows) x family (columns),
  # one line per posterior.

  om <- df %>%
    filter(
      grepl("^om_", param),
      scenario %in% c("dense", "sparse")
    )

  summary_df <- om %>%
    group_by(
      scenario,
      family,
      posterior,
      K
    ) %>%
    summarise(
      mean = mean(rel_bias_pct, na.rm = TRUE),
      sem = sd(rel_bias_pct, na.rm = TRUE) / sqrt(sum(!is.na(rel_bias_pct))),
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
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.2) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.6
    ) +
    scale_x_log10(
      breaks = sort(unique(summary_df$K)),
      labels = sort(unique(summary_df$K))
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

  n_scen <- length(unique(summary_df$scenario))
  n_fam <- length(unique(summary_df$family))

  savefig(
    p,
    "fig_phase1_grid",
    out_dir,
    width = 5.5 * n_fam,
    height = 4 * n_scen
  )
}


# -------------------------------------------------------------------------
# Nonlinear tier
# -------------------------------------------------------------------------

fig_nonlinear <- function(df, out_dir) {

  # Omega bias vs K for nonlinear (MM) tier,
  # one line per omega parameter.

  nl <- df

  if ("scenario" %in% names(df)) {
    nl <- df %>%
      filter(scenario == "nonlinear")
  }

  om <- nl %>%
    filter(grepl("^om_", param))

  summary_df <- om %>%
    group_by(param, K) %>%
    summarise(
      mean = mean(rel_bias_pct, na.rm = TRUE),
      sem = sd(rel_bias_pct, na.rm = TRUE) / sqrt(sum(!is.na(rel_bias_pct))),
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
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.2) +
    geom_errorbar(
      aes(
        ymin = mean - sem,
        ymax = mean + sem
      ),
      width = 0.08,
      linewidth = 0.6
    ) +
    scale_x_log10(
      breaks = sort(unique(summary_df$K)),
      labels = sort(unique(summary_df$K))
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


# -------------------------------------------------------------------------
# Real data
# -------------------------------------------------------------------------

fig_realdata <- function(datasets, out_dir) {

  # K=1 vs K=high grouped bar chart per omega parameter,
  # one panel per real dataset.

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

        k_lo <- min(df$K, na.rm = TRUE)
        k_hi <- max(df$K, na.rm = TRUE)

        df %>%
          filter(K %in% c(k_lo, k_hi)) %>%
          slice_head(n = 1, by = K) %>%
          select(K, all_of(om_cols)) %>%
          pivot_longer(
            cols = all_of(om_cols),
            names_to = "parameter",
            values_to = "estimate"
          ) %>%
          mutate(
            dataset = name,
            K_label = factor(
              paste0("K=", K),
              levels = c(
                paste0("K=", k_lo),
                paste0("K=", k_hi)
              )
            )
          )
      }
    )
  )

  p <- ggplot(
    plot_df,
    aes(
      x = parameter,
      y = estimate,
      fill = K_label
    )
  ) +
    geom_col(
      position = position_dodge(width = 0.8),
      width = 0.7
    ) +
    facet_wrap(
      ~dataset,
      nrow = 1,
      scales = "free_x"
    ) +
    labs(
      x = NULL,
      y = "omega estimate",
      fill = NULL
    ) +
    theme_publication()

  n <- length(datasets)

  savefig(
    p,
    "fig_realdata_k1_vs_khigh",
    out_dir,
    width = 5 * n,
    height = 4.2
  )
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

fig_deltaofv <- function(conditions, out_dir) {

  # Empirical dOFV histogram vs Self-Liang mixture reference density,
  # one panel per condition.

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

  xmax <- max(
    max(plot_df$dofv, na.rm = TRUE),
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
          density = 0.5 * dchisq(xx, df = 1)
        )
      }
    )
  )

  p <- ggplot(
    plot_df,
    aes(x = dofv)
  ) +
    geom_histogram(
      aes(y = after_stat(density)),
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
      ~condition,
      nrow = 1
    ) +
    labs(
      x = "dOFV",
      y = "density",
      colour = NULL
    ) +
    theme_publication()

  n <- length(conditions)

  savefig(
    p,
    "fig_deltaofv_calibration",
    out_dir,
    width = 6 * n,
    height = 4.5
  )
}


# -------------------------------------------------------------------------
# PSIS
# -------------------------------------------------------------------------

fig_psis <- function(df, out_dir) {

  # Per-subject ESS distributions by arm.

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

cat(strrep("=", 72), "\n", sep = "")
cat("MANUSCRIPT FIGURES\n")
cat(strrep("=", 72), "\n", sep = "")


# Phase 0 ---------------------------------------------------------------

df <- try_load(
  args$phase0_csv,
  "Phase 0"
)

if (!is.null(df)) {
  fig_phase0(df, args$out)
}


# Phase 1 ---------------------------------------------------------------

df <- try_load(
  args$phase1_csv,
  "Phase 1 grid"
)

if (!is.null(df)) {
  fig_phase1_grid(df, args$out)
}


# Nonlinear -------------------------------------------------------------

nonlinear_path <- if (!is.null(args$nonlinear_csv)) {
  args$nonlinear_csv
} else {
  args$phase1_csv
}

df <- try_load(
  nonlinear_path,
  "Phase 1 nonlinear"
)

if (!is.null(df)) {

  has_nonlinear <-
    "scenario" %in% names(df) &&
    "nonlinear" %in% df$scenario

  if (has_nonlinear) {

    fig_nonlinear(df, args$out)

  } else {

    message(
      "[skip] nonlinear figure: ",
      "no 'nonlinear' scenario rows in the given file"
    )
  }
}


# Real data -------------------------------------------------------------

realdata <- list()

df <- try_load(
  args$theoph_csv,
  "Theophylline"
)

if (!is.null(df)) {
  realdata[[length(realdata) + 1]] <- list(
    name = "theophylline",
    data = df
  )
}

df <- try_load(
  args$warfarin_csv,
  "warfarin"
)

if (!is.null(df)) {
  realdata[[length(realdata) + 1]] <- list(
    name = "warfarin",
    data = df
  )
}

if (length(realdata) > 0) {
  fig_realdata(realdata, args$out)
}


# dOFV calibration ------------------------------------------------------

calib <- list()

df <- try_load(
  args$deltaofv_free_csv,
  "dOFV free"
)

if (!is.null(df)) {
  calib[[length(calib) + 1]] <- list(
    name = "free",
    data = df
  )
}

df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV amortized"
)

if (!is.null(df)) {
  calib[[length(calib) + 1]] <- list(
    name = "amortized",
    data = df
  )
}

if (length(calib) > 0) {
  fig_deltaofv(calib, args$out)
}


# PSIS ------------------------------------------------------------------

df <- try_load(
  args$psis_csv,
  "PSIS"
)

if (!is.null(df)) {
  fig_psis(df, args$out)
}


cat(
  "\nDone. Figures written to ",
  args$out,
  "/\n",
  sep = ""
)