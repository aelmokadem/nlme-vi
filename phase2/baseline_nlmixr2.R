#!/usr/bin/env Rscript
# =============================================================================
# baseline_nlmixr2.R -- FOCEI / SAEM baseline fit, matching nlmevi_core's
# OneCmtOral exactly, for direct comparison against the VI results.
#
# INPUT CSV FORMAT:
#     ID, TIME, DV, AMT, EVID, CMT
#
# USAGE:
#     Rscript baseline_nlmixr2.R <input_csv> <output_csv> <foce|saem>
# =============================================================================

suppressPackageStartupMessages({
  library(nlmixr2)
  library(dplyr)
  library(readr)
})

# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop(
    "Usage: Rscript baseline_nlmixr2.R ",
    "<input_csv> <output_csv> <foce|saem>"
  )
}

input_csv  <- args[1]
output_csv <- args[2]
method     <- tolower(args[3])

if (!method %in% c("foce", "saem")) {
  stop("method must be 'foce' or 'saem'")
}


# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

dat <- read_csv(
  input_csv,
  show_col_types = FALSE,
  na = c("", "NA", ".")
)

n_subjects <- n_distinct(dat$ID)

cat(
  sprintf(
    "Loaded %d rows, %d subjects from %s\n",
    nrow(dat),
    n_subjects,
    input_csv
  )
)


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

one_cmt_oral <- function() {

  ini({

    lCL <- log(2.0)
    lV  <- log(20.0)
    lKa <- log(0.8)

    eta.CL ~ 0.09
    eta.V  ~ 0.09
    eta.Ka ~ 0.09

    add.sd <- 0.3
  })

  model({

    # ---------------------------------------------------------------------
    # ANALYTIC SOLUTION (linCmt()) -- matches VI's OneCmtOral exactly, which
    # evaluates this same linear one-compartment/first-order-absorption
    # system via its closed-form formula, not numerical integration. Using
    # the ODE form below for this model (which has a known closed form)
    # would make any runtime comparison unfair: rxode2 would be paying for
    # numerical integration at every likelihood evaluation during FOCEI's
    # linearization and SAEM's EM steps, on the R side only, for no reason
    # other than which solver happened to be picked -- inflating R's
    # reported runtime for a reason that has nothing to do with FOCEI/SAEM
    # as estimation algorithms. linCmt() auto-detects a one-compartment,
    # first-order-absorption model from the lowercase cl/v/ka parameter
    # names in scope -- this naming convention is the one point here that
    # should be verified on a small dataset (e.g. via baselines_r_direct)
    # before trusting a full run, same as every other accessor in this
    # script; nlmixr2's exact auto-detection rules can vary by version.
    # ---------------------------------------------------------------------
    cl <- exp(lCL + eta.CL)
    v  <- exp(lV  + eta.V)
    ka <- exp(lKa + eta.Ka)

    cp <- linCmt()
    logcp <- log(cp)

    logcp ~ add(add.sd)

    # ---------------------------------------------------------------------
    # ODE VERSION (commented out) -- numerically integrates the identical
    # linear system via rxode2's solver instead of the closed form above.
    # Parameter estimates from this form should match the analytic form
    # closely (same underlying math), but runtime will NOT match -- this
    # form is slower per likelihood evaluation and should only be used if
    # linCmt()'s auto-detection above turns out not to work for some
    # nlmixr2/rxode2 version, or if a future model extension (e.g. a
    # structure linCmt() can't represent) requires numerical integration.
    #
    # CL <- exp(lCL + eta.CL)
    # V  <- exp(lV  + eta.V)
    # Ka <- exp(lKa + eta.Ka)
    #
    # d/dt(depot)  <- -Ka * depot
    # d/dt(center) <- Ka * depot - (CL / V) * center
    #
    # cp <- center / V
    # logcp <- log(cp)
    #
    # logcp ~ add(add.sd)
    # ---------------------------------------------------------------------
  })
}


# -------------------------------------------------------------------------
# Fit + runtime measurement
# -------------------------------------------------------------------------

cat(
  sprintf(
    "\nRunning nlmixr2 %s fit...\n",
    toupper(method)
  )
)

# proc.time() returns a named vector: user.self, sys.self, elapsed (plus
# child-process components). elapsed alone is WALL-CLOCK time, vulnerable
# to the same sleep/contention noise documented throughout this project's
# Python-side timing (see nlme_vi_phase2_baselines.py). user.self + sys.self
# is the CPU-time equivalent of Python's time.process_time(), immune to
# that noise -- that is the quantity nlme_vi_phase2_baselines.py expects,
# in a column named cpu_secs, for the fair apples-to-apples comparison
# against VI's own cpu_secs. wall_secs is also written, for context only,
# matching the sibling convention used on the Python/VI side.
start_time <- proc.time()

if (method == "foce") {

  fit <- nlmixr2(
    one_cmt_oral(),
    dat,
    est = "foce",
    control = foceiControl(
      print = 0
    )
  )

} else {

  fit <- nlmixr2(
    one_cmt_oral(),
    dat,
    est = "saem",
    control = saemControl(
      print = 0,
      nBurn = 200,
      nEm = 300
    )
  )

}

end_time <- proc.time()

cpu_secs <- unname(
  (end_time["user.self"] + end_time["sys.self"]) -
  (start_time["user.self"] + start_time["sys.self"])
)

wall_secs <- unname(
  end_time["elapsed"] - start_time["elapsed"]
)

cat(
  sprintf(
    "%s fit completed in %.3f CPU seconds (%.3f wall seconds)\n",
    toupper(method),
    cpu_secs,
    wall_secs
  )
)


# -------------------------------------------------------------------------
# Extract fixed effects
# -------------------------------------------------------------------------

pop <- fixef(fit)

get_fixef <- function(name) {

  if (name %in% names(pop)) {
    as.numeric(pop[[name]])
  } else {
    NA_real_
  }
}


# -------------------------------------------------------------------------
# Extract omega
# -------------------------------------------------------------------------

om <- tryCatch(
  as.matrix(fit$omega),
  error = function(e) NULL
)

get_omega_sd <- function(name) {

  if (is.null(om)) {
    return(NA_real_)
  }

  if (
    name %in% rownames(om) &&
    name %in% colnames(om)
  ) {

    value <- om[name, name]

    if (
      is.finite(value) &&
      value >= 0
    ) {
      return(sqrt(value))
    }
  }

  NA_real_
}


# -------------------------------------------------------------------------
# Extract residual SD
# -------------------------------------------------------------------------

get_sigma <- function(fit) {

  # First try fixed-effect representation.
  fx <- tryCatch(
    fixef(fit),
    error = function(e) NULL
  )

  if (!is.null(fx)) {

    candidates <- c(
      "add.sd",
      "add.sd.",
      "sigma"
    )

    for (nm in candidates) {

      if (nm %in% names(fx)) {

        value <- suppressWarnings(
          as.numeric(fx[[nm]])
        )

        if (
          length(value) == 1 &&
          is.finite(value)
        ) {
          return(value)
        }
      }
    }
  }

  # Fallback: theta.
  th <- tryCatch(
    fit$theta,
    error = function(e) NULL
  )

  if (!is.null(th)) {

    candidates <- c(
      "add.sd",
      "add.sd.",
      "sigma"
    )

    for (nm in candidates) {

      if (nm %in% names(th)) {

        value <- suppressWarnings(
          as.numeric(th[[nm]])
        )

        if (
          length(value) == 1 &&
          is.finite(value)
        ) {
          return(value)
        }
      }
    }
  }

  NA_real_
}


# -------------------------------------------------------------------------
# OFV / log likelihood
# -------------------------------------------------------------------------

get_ofv <- function(fit) {

  candidates <- list(
    function() fit$objDf$OBJF[1],
    function() fit$objDf$objective[1],
    function() -2 * as.numeric(logLik(fit))
  )

  for (fun in candidates) {

    value <- tryCatch(
      as.numeric(fun()),
      error = function(e) NA_real_
    )

    if (
      length(value) == 1 &&
      is.finite(value)
    ) {
      return(value)
    }
  }

  NA_real_
}


get_loglik <- function(fit) {

  value <- tryCatch(
    as.numeric(logLik(fit)),
    error = function(e) NA_real_
  )

  if (
    length(value) == 1 &&
    is.finite(value)
  ) {
    return(value)
  }

  NA_real_
}


# -------------------------------------------------------------------------
# Result
# -------------------------------------------------------------------------

result <- tibble(

  method = method,

  CL = exp(
    get_fixef("lCL")
  ),

  V = exp(
    get_fixef("lV")
  ),

  ka = exp(
    get_fixef("lKa")
  ),

  om_CL = get_omega_sd(
    "eta.CL"
  ),

  om_V = get_omega_sd(
    "eta.V"
  ),

  om_ka = get_omega_sd(
    "eta.Ka"
  ),

  sigma = get_sigma(
    fit
  ),

  ofv = get_ofv(
    fit
  ),

  logLik = get_loglik(
    fit
  ),

  # IMPORTANT: column names must match what nlme_vi_phase2_baselines.py
  # reads back via getattr(r, "cpu_secs", ...) / getattr(r, "wall_secs", ...).
  # cpu_secs is the primary quantity for any speed comparison; wall_secs
  # is context only (see the comment above start_time for why).
  cpu_secs = cpu_secs,
  wall_secs = wall_secs,

  n_subjects = n_subjects
)


# -------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------

dir.create(
  dirname(output_csv),
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  result,
  output_csv
)

cat(
  sprintf(
    "\nWrote result -> %s\n",
    output_csv
  )
)

print(
  result,
  width = Inf
)