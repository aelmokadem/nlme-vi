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

    CL <- exp(lCL + eta.CL)
    V  <- exp(lV  + eta.V)
    Ka <- exp(lKa + eta.Ka)

    d/dt(depot)  <- -Ka * depot
    d/dt(center) <- Ka * depot - (CL / V) * center

    cp <- center / V
    logcp <- log(cp)

    logcp ~ add(add.sd)
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

# proc.time()[["elapsed"]] measures wall-clock elapsed time.
start_time <- proc.time()[["elapsed"]]

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

end_time <- proc.time()[["elapsed"]]

runtime_s <- as.numeric(
  end_time - start_time
)

cat(
  sprintf(
    "%s fit completed in %.3f seconds\n",
    toupper(method),
    runtime_s
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

  # IMPORTANT:
  # Use a standard runtime column name for downstream comparison tables.
  runtime_s = runtime_s,

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