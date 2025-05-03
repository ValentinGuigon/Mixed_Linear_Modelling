##### VARIABLES #####

# Free memory
rm(list = ls(all.names = TRUE)) # Clear all objects including hidden ones.
gc() # Free up memory and report the memory usage.

# Load required packages
required_packages <- c("lme4", "pracma", "ggplot2", "sjPlot", "emmeans", "interactions", "jtools",
                       "ggpubr", "mediation", "RColorBrewer", "broom.mixed", "purrr",
                       "rprojroot", "webshot")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)
lapply(required_packages, require, character.only = TRUE)

# Generate sample data
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

# 1. Total effect of Covariate1 on Outcome
fit_total_effect <- glm(Outcome ~ Covariate1 * FactorA + (1 | Subject), data = sample_data, family = binomial(link = "logit"))
fit_total_effect_confint <- confint(fit_total_effect) # confint takes forever to compute

# 2. Mediator effect: Covariate1 on Covariate2
fit_mediator <- lm(Covariate2 ~ Covariate1 * FactorA + (1 | Subject), data = sample_data)
fit_mediator_confint <- confint(fit_mediator) # confint takes forever to compute

# 3. Direct effect: Covariate2 on Outcome
fit_dv <- glm(Outcome ~ Covariate2 + Covariate1 * FactorA + (1 | Subject), data = sample_data, family = binomial(link = "logit"))
fit_dv_confint <- confint(fit_dv) # confint takes forever to compute

# 4. Mediation analysis
results <- mediation::mediate(fit_mediator, fit_dv, treat = "Covariate1", mediator = "Covariate2", boot = TRUE, sims = 100)
summary(results)
plot(results)

results2 <- mediation::mediate(fit_mediator, fit_dv, treat = "Covariate1", mediator = "Covariate2", boot = TRUE, sims = 100)
summary(results2)
plot(results2)
