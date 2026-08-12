#!/usr/bin/env Rscript
# =============================================================================
# baseline_nlmixr2.R -- FOCEI / SAEM baseline fit, matching nlmevi_core's
# OneCmtOral exactly, for direct comparison against the VI results.
#
# *** THIS SCRIPT HAS NOT BEEN EXECUTED OR VALIDATED. ***
# The sandbox this project was built in has no R installation and no network
# access to CRAN, so this could not be run or tested end to end. It follows
# the documented nlmixr2 API as of early 2026 as closely as possible, but
# accessor names on the fit object (fixef(), $omega, $theta, $objDf) have
# changed across nlmixr2 versions before and may need small adjustments for
# whatever version is installed. Before trusting any output: run it on a
# toy dataset first and eyeball print(fit) to confirm the accessors below
# match what's actually on the object.
#
# MODEL -- must match OneCmtOral in nlmevi_core.py exactly for the
# comparison to be meaningful:
#     log(CL) = lCL + eta.CL,  log(V) = lV + eta.V,  log(Ka) = lKa + eta.Ka
#     Y = log(concentration) ~ Normal(0, add.sd^2)   [additive error on log
#     scale -- this is the log-normal residual error VI's generative model
#     uses, NOT nlmixr2's default proportional/combined error on the
#     natural scale. Do not swap in a different error model without also
#     changing nlmevi_core's OneCmtOral.log_lik to match, or FOCE/SAEM and
#     VI will be fitting different statistical models and the comparison
#     is meaningless even if both converge cleanly.]
#
# INPUT CSV FORMAT (written by nlme_vi_phase2_baselines.py):
#     ID, TIME, DV, AMT, EVID, CMT
#     One dosing row per subject (TIME=0, AMT=dose, EVID=1, DV=.), then one
#     row per observation (DV = log concentration, AMT=0, EVID=0).
#
# USAGE
#     Rscript baseline_nlmixr2.R <input_csv> <output_csv> <foce|saem>
# =============================================================================

suppressPackageStartupMessages({
  library(nlmixr2)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript baseline_nlmixr2.R <input_csv> <output_csv> <foce|saem>")
}
input_csv  <- args[1]
output_csv <- args[2]
method     <- tolower(args[3])
if (!method %in% c("foce", "saem")) {
  stop("method must be 'foce' or 'saem'")
}

dat <- read_csv(input_csv, show_col_types = FALSE)
cat(sprintf("Loaded %d rows, %d subjects from %s\n",
            nrow(dat), length(unique(dat$ID)), input_csv))

# -----------------------------------------------------------------------
# Model definition. Initial values are deliberately off-truth (matching
# the VI side's theta_init), same reasoning: recovering the true values is
# a real test of the estimation method, not a fixed point of the starting
# guess.
# -----------------------------------------------------------------------
one_cmt_oral <- function() {
  ini({
    lCL <- log(2.0)      # off-truth start, matches VI's theta_init
    lV  <- log(20.0)
    lKa <- log(0.8)
    eta.CL ~ 0.09         # (0.3)^2 -- off-truth start
    eta.V  ~ 0.09
    eta.Ka ~ 0.09
    add.sd <- 0.3
  })
  model({
    CL <- exp(lCL + eta.CL)
    V  <- exp(lV  + eta.V)
    Ka <- exp(lKa + eta.Ka)
    d/dt(depot)  <- -Ka * depot
    d/dt(center) <-  Ka * depot - (CL / V) * center
    cp <- center / V
    logcp <- log(cp)
    logcp ~ add(add.sd)
  })
}

# -----------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------
t0 <- Sys.time()
if (method == "foce") {
  fit <- nlmixr2(one_cmt_oral(), dat, est = "foce",
                 control = foceiControl(print = 0))
} else {
  fit <- nlmixr2(one_cmt_oral(), dat, est = "saem",
                 control = saemControl(print = 0, nBurn = 200, nEm = 300))
}
elapsed <- as.numeric(Sys.time() - t0, units = "secs")
cat(sprintf("Fit completed in %.1fs\n", elapsed))

# -----------------------------------------------------------------------
# Extract results onto the SAME scale/columns the VI side reports, so the
# two can be merged directly:  CL, V, ka, om_CL, om_V, om_ka, sigma
#
# *** UNVALIDATED ACCESSORS -- see warning at top of file ***
# -----------------------------------------------------------------------
pop <- fixef(fit)
om  <- as.matrix(fit$omega)

get_theta <- function(fit, name, default = NA) {
  tryCatch(as.numeric(fit$theta[name]), error = function(e) default)
}

result <- data.frame(
  method    = method,
  CL        = exp(unname(pop["lCL"])),
  V         = exp(unname(pop["lV"])),
  ka        = exp(unname(pop["lKa"])),
  om_CL     = sqrt(om["eta.CL", "eta.CL"]),
  om_V      = sqrt(om["eta.V",  "eta.V"]),
  om_ka     = sqrt(om["eta.Ka", "eta.Ka"]),
  sigma     = get_theta(fit, "add.sd"),
  ofv       = tryCatch(fit$objDf$OBJF[1], error = function(e) NA),
  logLik    = tryCatch(as.numeric(logLik(fit)), error = function(e) NA),
  elapsed_s = elapsed,
  n_subjects = length(unique(dat$ID))
)

write_csv(result, output_csv)
cat(sprintf("Wrote result -> %s\n", output_csv))
print(result)
