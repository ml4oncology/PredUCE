# Load necessary libraries
library(lme4)
library(reticulate)


# GLMM function
GLMM <- function(data, response, predictor, nAGQ) {
  
  # Rescale predictors if necessary
  data$Time <- scale(data$Time)
  
  # Construct the formula
  formula <- as.formula(paste(response, "~", predictor, "+ Time + (1 | PatientID)"))
  
  # Fit the model with additional control parameters
  start_time <- Sys.time()
  model <- glmer(formula, family = binomial, data = data, nAGQ = nAGQ, control = control)
  end_time <- Sys.time()
  
  # Check for convergence
  if (!all(model@optinfo$conv$lme4$conv)) {
    warning("Model did not converge")
  }
  
  # Print timing
  print(end_time - start_time)
  
  # Return the model summary
  summary(model)
}
