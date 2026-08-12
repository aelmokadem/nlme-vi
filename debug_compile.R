library(nlmixr2)

dat <- readr::read_csv("outputs/phase2_baselines/rep0_data.csv",
                        na = c("", "NA", "."), show_col_types = FALSE)

mod <- function() {
  ini({
    lCL <- log(2.0); lV <- log(20.0); lKa <- log(0.8)
    eta.CL ~ 0.09; eta.V ~ 0.09; eta.Ka ~ 0.09
    add.sd <- 0.3
  })
  model({
    CL <- exp(lCL + eta.CL)
    V  <- exp(lV + eta.V)
    Ka <- exp(lKa + eta.Ka)
    d/dt(depot)  <- -Ka * depot
    d/dt(center) <-  Ka * depot - (CL / V) * center
    cp <- center / V
    logcp <- log(cp)
    logcp ~ add(add.sd)
  })
}

tryCatch({
  fit <- nlmixr2(mod(), dat, est = "foce")
  cat("FIT SUCCEEDED\n")
}, error = function(e) {
  cat("CAUGHT ERROR:", conditionMessage(e), "\n")
})

cat("\n\n===== rxLastCompile() output =====\n")
print(rxode2::rxLastCompile())
