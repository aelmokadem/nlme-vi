# publication/make_figures.R
#
# Manuscript figures from the project's result CSVs.
#
# Companion to make_tables.R. Input paths are explicit because several source
# scripts may overwrite fixed default filenames across different conditions.
# This script therefore never guesses which file corresponds to which
# condition.
#
# Figures deliberately have NO captions or descriptive titles baked into the
# image. Only axis labels, legends, and panel labels needed for interpretation
# are included. Caption text belongs in the manuscript.
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

library(tidyverse)
library(optparse)
library(here)
library(patchwork)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

try_load <- function(path, label) {

  if (is.null(path) || is.na(path) || path == "") {
    message("[skip] ", label, ": no path given")
    return(NULL)
  }

  full_path <- if (fs::is_absolute_path(path)) {
    path
  } else {
    here(path)
  }

  if (!file.exists(full_path)) {
    message("[skip] ", label, ": ", full_path, " not found")
    return(NULL)
  }

  df <- readr::read_csv(
    full_path,
    show_col_types = FALSE
  )

  message(
    "[loaded] ", label, ": ",
    full_path,
    " (", nrow(df), " rows)"
  )

  df
}


savefig <- function(plot, name, out_dir, width = 7, height = 5) {

  full_out_dir <- if (fs::is_absolute_path(out_dir)) {
    out_dir
  } else {
    here(out_dir)
  }

  dir.create(
    full_out_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  path <- file.path(
    full_out_dir,
    paste0(name, ".png")
  )

  ggsave(
    filename = path,
    plot = plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )

  message("-> ", path)
}


sem <- function(x) {

  n <- sum(!is.na(x))

  if (n <= 1) {
    return(NA_real_)
  }

  sd(x, na.rm = TRUE) / sqrt(n)
}


theme_manuscript <- function() {

  theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        linewidth = 0.3
      ),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold"),
      legend.title = element_blank()
    )
}


# -------------------------------------------------------------------------
# Phase 0
# -------------------------------------------------------------------------

fig_phase0 <- function(df, out_dir) {

  # Omega bias vs K, one panel per posterior.

  om <- df %>%
    filter(
      str_starts(param, "om_")
    )

  if (nrow(om) == 0) {
    message("[skip] Phase 0 figure: no omega parameters found")
    return(invisible(NULL))
  }

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

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean_bias,
      group = param,
      shape = param,
      linetype = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_line() +
    geom_point(size = 2.5) +
    geom_errorbar(
      aes(
        ymin = mean_bias - sem_bias,
        ymax = mean_bias + sem_bias
      ),
      width = 0
    ) +
    scale_x_continuous(
      trans = scales::transform_log(base = 2),
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

  n_post <- n_distinct(summary_df$posterior)

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

  # Omega bias vs K.
  #
  # Rows = scenario
  # Columns = posterior family
  # Lines = posterior parameterization/method

  required_cols <- c(
    "param",
    "scenario",
    "family",
    "posterior",
    "K",
    "rel_bias_pct"
  )

  missing_cols <- setdiff(
    required_cols,
    names(df)
  )

  if (length(missing_cols) > 0) {
    message(
      "[skip] Phase 1 grid: missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
    return(invisible(NULL))
  }

  om <- df %>%
    filter(
      str_starts(param, "om_"),
      scenario %in% c("dense", "sparse")
    )

  if (nrow(om) == 0) {
    message(
      "[skip] Phase 1 grid: ",
      "no dense/sparse omega rows found"
    )
    return(invisible(NULL))
  }

  # Diagnostic so we can verify all intended families are actually present.
  message("Phase 1 grid combinations found:")

  print(
    om %>%
      count(
        scenario,
        family,
        posterior,
        name = "n"
      ) %>%
      arrange(
        scenario,
        family,
        posterior
      )
  )

  message(
    "Phase 1 families found: ",
    paste(
      sort(unique(om$family)),
      collapse = ", "
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

  # Explicitly preserve every family that actually exists in the input.
  #
  # This avoids silent dropping/reordering of families such as "gaussian".
  family_levels <- sort(
    unique(as.character(om$family))
  )

  posterior_levels <- sort(
    unique(as.character(om$posterior))
  )

  summary_df <- summary_df %>%
    mutate(
      scenario = factor(
        scenario,
        levels = c(
          "dense",
          "sparse"
        )
      ),
      family = factor(
        family,
        levels = family_levels
      ),
      posterior = factor(
        posterior,
        levels = posterior_levels
      )
    )

  p <- ggplot(
    summary_df,
    aes(
      x = K,
      y = mean_bias,
      group = posterior,
      shape = posterior,
      linetype = posterior
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_line() +
    geom_point(
      size = 2.5
    ) +
    geom_errorbar(
      aes(
        ymin = mean_bias - sem_bias,
        ymax = mean_bias + sem_bias
      ),
      width = 0
    ) +
    scale_x_continuous(
      trans = scales::transform_log(base = 2),
      breaks = sort(
        unique(summary_df$K)
      )
    ) +
    facet_grid(
      rows = vars(scenario),
      cols = vars(family),
      drop = FALSE
    ) +
    labs(
      x = "K",
      y = "omega bias (%)"
    ) +
    theme_manuscript()

  n_families <- length(
    family_levels
  )

  n_scenarios <- n_distinct(
    summary_df$scenario
  )

  savefig(
    p,
    "fig_phase1_grid",
    out_dir,
    width = 5.5 * n_families,
    height = 4 * n_scenarios
  )
}


# -------------------------------------------------------------------------
# Phase 1 nonlinear
# -------------------------------------------------------------------------

fig_nonlinear <- function(df, out_dir) {

  # Omega bias vs K for the nonlinear tier,
  # one line per omega parameter.

  if (
    "scenario" %in% names(df)
  ) {

    nl <- df %>%
      filter(
        scenario == "nonlinear"
      )

  } else {

    nl <- df

  }

  om <- nl %>%
    filter(
      str_starts(param, "om_")
    )

  if (nrow(om) == 0) {
    message(
      "[skip] nonlinear figure: ",
      "no omega rows found"
    )
    return(invisible(NULL))
  }

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
      shape = param,
      linetype = param
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
    ) +
    geom_line() +
    geom_point(
      size = 2.5
    ) +
    geom_errorbar(
      aes(
        ymin = mean_bias - sem_bias,
        ymax = mean_bias + sem_bias
      ),
      width = 0
    ) +
    scale_x_continuous(
      trans = scales::transform_log(base = 2),
      breaks = sort(
        unique(summary_df$K)
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


# -------------------------------------------------------------------------
# Real-data comparison
# -------------------------------------------------------------------------

fig_realdata <- function(datasets, out_dir) {

  # K = lowest vs K = highest.
  # Grouped bar chart per omega parameter,
  # one panel per real dataset.

  plot_df <- purrr::imap_dfr(
    datasets,
    function(df, dataset_name) {

      if (!"K" %in% names(df)) {
        message(
          "[skip] ", dataset_name,
          ": no K column"
        )
        return(tibble())
      }

      om_cols <- names(df) %>%
        keep(
          ~ str_starts(.x, "om_")
        )

      if (length(om_cols) == 0) {
        message(
          "[skip] ", dataset_name,
          ": no omega columns"
        )
        return(tibble())
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
        group_by(K) %>%
        slice(1) %>%
        ungroup() %>%
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
          K_num = K,
          K = paste0(
            "K=",
            K
          )
        )
    }
  )

  if (nrow(plot_df) == 0) {
    message(
      "[skip] real-data figure: ",
      "no usable omega columns found"
    )
    return(invisible(NULL))
  }

  k_levels <- plot_df %>%
    distinct(
      K,
      K_num
    ) %>%
    arrange(
      K_num
    ) %>%
    pull(K)

  plot_df <- plot_df %>%
    mutate(
      K = factor(
        K,
        levels = k_levels
      )
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
      y = "omega estimate",
      fill = NULL
    ) +
    theme_manuscript() +
    theme(
      panel.grid.major.x = element_blank()
    )

  n_datasets <- n_distinct(
    plot_df$dataset
  )

  savefig(
    p,
    "fig_realdata_k1_vs_khigh",
    out_dir,
    width = 5 * n_datasets,
    height = 4.2
  )
}


# -------------------------------------------------------------------------
# dOFV calibration
# -------------------------------------------------------------------------

fig_deltaofv <- function(conditions, out_dir) {

  # Empirical dOFV histogram vs Self-Liang
  # 0.5 * chi-square(1) reference density.

  plot_df <- purrr::imap_dfr(
    conditions,
    function(df, label) {

      if (!"dofv" %in% names(df)) {

        message(
          "[skip] dOFV ", label,
          ": no dofv column"
        )

        return(tibble())
      }

      df %>%
        transmute(
          condition = label,
          dofv = dofv
        )
    }
  )

  if (nrow(plot_df) == 0) {
    message(
      "[skip] dOFV figure: ",
      "no observations found"
    )
    return(invisible(NULL))
  }

  max_x <- max(
    max(
      plot_df$dofv,
      na.rm = TRUE
    ),
    8
  )

  reference_df <- tibble(
    x = seq(
      0.01,
      max_x,
      length.out = 300
    ),
    density = 0.5 * dchisq(
      x,
      df = 1
    )
  )

  reference_df <- tidyr::crossing(
    condition = unique(
      plot_df$condition
    ),
    reference_df
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
      data = reference_df,
      aes(
        x = x,
        y = density
      ),
      inherit.aes = FALSE,
      linewidth = 1
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5
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

  n_conditions <- n_distinct(
    plot_df$condition
  )

  savefig(
    p,
    "fig_deltaofv_calibration",
    out_dir,
    width = 6 * n_conditions,
    height = 4.5
  )
}


# -------------------------------------------------------------------------
# PSIS ESS
# -------------------------------------------------------------------------

fig_psis <- function(df, out_dir) {

  # Per-subject ESS distribution by arm.

  required_cols <- c(
    "arm",
    "ess"
  )

  if (
    !all(
      required_cols %in% names(df)
    )
  ) {

    message(
      "[skip] PSIS figure: ",
      "required columns 'arm' and 'ess' not found"
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
    theme_manuscript()

  savefig(
    p,
    "fig_psis_ess",
    out_dir,
    width = 6,
    height = 4.5
  )
}


# -------------------------------------------------------------------------
# Command-line options
# -------------------------------------------------------------------------

option_list <- list(

  make_option(
    "--phase0-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--phase1-csv",
    type = "character",
    default = NULL
  ),

  make_option(
    "--nonlinear-csv",
    type = "character",
    default = NULL
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
    "--out",
    type = "character",
    default = "publication/figures"
  )
)


args <- parse_args(
  OptionParser(
    option_list = option_list
  )
)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

cat(
  paste0(
    rep("=", 72),
    collapse = ""
  ),
  "\n"
)

cat(
  "MANUSCRIPT FIGURES\n"
)

cat(
  paste0(
    rep("=", 72),
    collapse = ""
  ),
  "\n"
)


# Phase 0 ----------------------------------------------------------------

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


# Phase 1 grid ------------------------------------------------------------

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


# Phase 1 nonlinear -------------------------------------------------------

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
  "Phase 1 nonlinear"
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

    fig_nonlinear(
      df_nonlinear,
      args$out
    )

  } else {

    message(
      "[skip] nonlinear figure: ",
      "no 'nonlinear' scenario rows ",
      "in the given file"
    )

  }

}


# Real data ---------------------------------------------------------------

realdata <- list()

df <- try_load(
  args$theoph_csv,
  "Theophylline"
)

if (!is.null(df)) {

  realdata$theophylline <- df

}

df <- try_load(
  args$warfarin_csv,
  "Warfarin"
)

if (!is.null(df)) {

  realdata$warfarin <- df

}

if (length(realdata) > 0) {

  fig_realdata(
    realdata,
    args$out
  )

}


# dOFV calibration --------------------------------------------------------

calib <- list()

df <- try_load(
  args$deltaofv_free_csv,
  "dOFV free"
)

if (!is.null(df)) {

  calib$free <- df

}

df <- try_load(
  args$deltaofv_amortized_csv,
  "dOFV amortized"
)

if (!is.null(df)) {

  calib$amortized <- df

}

if (length(calib) > 0) {

  fig_deltaofv(
    calib,
    args$out
  )

}


# PSIS -------------------------------------------------------------------

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


# Done -------------------------------------------------------------------

full_out_dir <- if (
  fs::is_absolute_path(args$out)
) {

  args$out

} else {

  here(args$out)

}

cat(
  "\nDone. Figures written to ",
  full_out_dir,
  "/\n",
  sep = ""
)