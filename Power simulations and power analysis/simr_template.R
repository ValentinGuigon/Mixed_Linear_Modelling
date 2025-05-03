# For extend():
# object = fitted model (REQUIRED)
# along = variable to extend (e.g. "Subject") (REQUIRED)
# n = target levels (REQUIRED)
# within = grouping structure (optional)
# values = custom level names (optional)

# For powerSim()/powerCurve():
# test = effect to test (REQUIRED):
#   - fixed("term") for single terms
#   - fixed("term","z") for Z-tests
#   - fixed("term","lr") for likelihood ratio
#   - fcompare(~reduced_formula) for model comparison
# nsim = iterations (DEFAULT=100)
# alpha = threshold (DEFAULT=0.05)
# progress = show bar (DEFAULT=TRUE)

# Error workaround (from simr#203: https://github.com/pitakakariki/simr/issues/203):
# test_fixed <- fixed("Term")
# attr(test_fixed, "text") <- function(...) NULL

## Memory management
rm(list = ls(all.names = TRUE)) # Clear all objects including hidden ones.
gc() # Free up memory and report the memory usage.

## Load packages
required_packages <- c("lme4", "magrittr", "dplyr", "simr", "rprojroot")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)
lapply(required_packages, require, character.only = TRUE)

## Generate sample data
set.seed(123)
num_subjects <- 100  # Number of subjects
sample_data <- data.frame(
  Subject = factor(rep(1:num_subjects, each = 10)),
  Outcome = rbinom(10 * num_subjects, 1, 0.5),
  FactorA = factor(sample(c("Level1", "Level2"), 10 * num_subjects, replace = TRUE)),
  FactorB = factor(sample(c("Level1", "Level2"), 10 * num_subjects, replace = TRUE)),
  Covariate1 = rnorm(10 * num_subjects),
  Covariate2 = rnorm(10 * num_subjects),
  Covariate3 = rnorm(10 * num_subjects),
  Covariate4 = rnorm(10 * num_subjects),
  Age = rnorm(10 * num_subjects, mean = 30, sd = 5),
  Sex = factor(sample(c("Male", "Female"), 10 * num_subjects, replace = TRUE)),
  Education = factor(sample(c("HighSchool", "Bachelors", "Masters", "PhD"), 10 * num_subjects, replace = TRUE)),
  RandomFactor = factor(rep(1:num_subjects, each = 10))
)

## Compute GLM
full_model <- glmer(
  Outcome ~ FactorA * Covariate1 + FactorB * Covariate2 + Covariate3 + FactorA * Covariate4 +
    Age + Sex + Education + Covariate1 +
    (1 | Subject),
  data = sample_data, family = binomial(link = 'logit'),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)
summary(full_model)

## Compute observed effect sizes
# Observed effect size for Covariate1
doTest(full_model, fixed("Covariate1", "z"))

# Observed effect size for Covariate2
doTest(full_model, fixed("Covariate2", "z"))

## Compute actual power for simple effect of Covariate1
power_sim_covariate1 <- powerSim(full_model, test = fixed("Covariate1"), nsim = 100, alpha = 0.05, progress = TRUE)

## Compute actual power for interaction effect of Covariate1
power_sim_interaction_covariate1 <- powerSim(full_model, test = fixed("FactorALevel2:Covariate1"), nsim = 100, alpha = 0.05, progress = TRUE)

## Compute actual power for simple effect of Covariate2
power_sim_covariate2 <- powerSim(full_model, test = fixed("Covariate2"), nsim = 100, alpha = 0.05, progress = TRUE)

## Compute actual power for interaction effect of Covariate2
power_sim_interaction_covariate2 <- powerSim(full_model, test = fixed("FactorBLevel2:Covariate2"), nsim = 100, alpha = 0.05, progress = TRUE)

## Model Simplification
# Create a simplified version of the model
simplified_model <- glmer(
  Outcome ~ FactorA * Covariate1 + FactorB * Covariate2 + Covariate3 + FactorA * Covariate4 +
    (1 | Subject),
  data = sample_data, family = binomial(link = 'logit'),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)
summary(simplified_model)

## Fixed Effects Manipulation
# Adjust the fixed effects of the simplified model
adjusted_simplified_model <- simplified_model
fixef(adjusted_simplified_model)['Covariate1'] <- 0.1  # Example adjustment

## Power Curve Analysis
# Extend the number of subjects for the adjusted model
extended_adjusted_model <- extend(adjusted_simplified_model, along = "Subject", n = 250)

# Plot the power curve for the main effect of Covariate1
power_curve_covariate1 <- powerCurve(
  extended_adjusted_model, test = fixed("Covariate1"), along = "Subject",
  breaks = c(80, 150, 200, 250), nsim = 100, alpha = 0.05, progress = TRUE
)
plot(power_curve_covariate1)
print(power_curve_covariate1$errors)
print(power_curve_covariate1$warnings)
print(power_curve_covariate1)

# Plot the power curve for the main effect of Covariate2
power_curve_covariate2 <- powerCurve(
  extended_adjusted_model, test = fixed("Covariate2"), along = "Subject",
  breaks = c(80, 150, 200, 250), nsim = 100, alpha = 0.05, progress = TRUE
)
plot(power_curve_covariate2)
print(power_curve_covariate2$errors)
print(power_curve_covariate2$warnings)
print(power_curve_covariate2)

## Model Comparison
# Compare the full model with the simplified model
power_sim_model_comparison <- powerSim(full_model, test = fcompare(Outcome ~ FactorA * Covariate1 + FactorB * Covariate2 + Covariate3 + FactorA * Covariate4 + (1 | Subject)), nsim = 100, alpha = 0.05, progress = TRUE)

# Extend the number of subjects for the full model
extended_full_model <- extend(full_model, along = "Subject", n = 250)

# Plot the power curve for the model comparison
power_curve_model_comparison <- powerCurve(
  extended_full_model, test = fcompare(Outcome ~ FactorA * Covariate1 + FactorB * Covariate2 + Covariate3 + FactorA * Covariate4 + (1 | Subject)),
  along = "Subject", breaks = c(80, 150, 200, 250), nsim = 100, alpha = 0.05, progress = TRUE
)
plot(power_curve_model_comparison)
print(power_curve_model_comparison$errors)
print(power_curve_model_comparison$warnings)
print(power_curve_model_comparison)
