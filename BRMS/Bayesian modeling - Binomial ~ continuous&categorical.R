### Predict binomial y with continuous and/or categorical x ###


## Packages

# install.packages(c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "BEST", "coda", "LearnBayes", "markdown", "mcmc", "MCMCpack", "MuMIn", "reshape2", "rmarkdown", "brms", "tidyverse", "tidybayes", "bayesplot", "shinystan", "lme4", "patchwork"), dependencies = TRUE)
# install.packages("ks")

# my_packages <- c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "coda", "LearnBayes", "mcmc", "MCMCpack", "MuMIn", "reshape2", "tidybayes", "bayesplot", "shinystan", "patchwork")
# lapply(my_packages, require, character.only = TRUE)   

# best practice: install R, Rstudio and JAGS in the same folder
# download and install on windows: JAGS-4.2.0-Rtools33 or more
# install.packages("rjags") # In case of trouble, write in C:\Users\vguigon\Documents\.R\Makevars.win the following: JAGS_HOME=C:\programs\JAGS\JAGS-4.0.0
# install.packages("BEST")

library(brms) 
library(modelr)
library(rethinking)
library(ks)
library(tidyverse)
library(tidybayes)
library(modelr)
library(mcmc)
library(bayesplot)
library(bayestestR)
library(see)
library(logspline)
library(rjags)
library(BEST)

# More functions at: https://mc-stan.org/rstanarm/reference/index.html
# Graphical posterior predictive checks: https://mc-stan.org/rstanarm/reference/pp_check.stanreg.html


## Data
d1 <- read.csv("titanic.csv")
head(d1, 10)


# y ~ x1 + x2
d1 %>%
  group_by(pclass, gender) %>%
  summarise(p = mean(survival) ) %>%
  ggplot(aes(x = as.factor(pclass), y = p, fill = as.factor(gender) ) ) +
  geom_bar(position = position_dodge(0.5), stat = "identity", alpha = 0.8) +
  xlab("class") + ylab("p(survival)")


# multi-level 
d1 %>%
  gather(pclass, age, 3:4) %>%
  ggplot(aes(x = age, y = survival, colour = pclass, fill = pclass) ) +
  geom_point(pch = 21, size = 4, color = "white", alpha = 1) +
  stat_smooth(method = "lm", fullrange = TRUE) +
  facet_wrap(~ pclass)


# centering and standardising predictors
d1 <-
  d1 %>%
  mutate(
    pclass = ifelse(pclass == "lower", -0.5, 0.5),
    gender = ifelse(gender == "female", -0.5, 0.5),
    age = scale(age) %>% as.numeric,
    parch = scale(parch) %>% as.numeric
  )

# IN CASE : rescaling predictors (e.g.)
# d1$gender <- ifelse(d1$gender == "F", -0.5, 0.5)                        # reweight binomial variable 'gender'
# d1$mother <- scale(d1$mother) %>% as.numeric                            # rescale variable: calculate mean+SD of the vector then scale each element by subtracting mean and dividing by SD
# d1 <- d1 %>% mutate(mother.s = (mother - mean(mother) ) / sd(mother) )  # in case of standardization
# d1 <- d1 %>% mutate(mother.c = (mother - mean(mother) )                 # in case of data to center





## I. Set models 

# [multi-level requires repeated measures] (# backend = "cmdstanr" -> if convergence is too slow or too buggy)

# priors = ~ 1 -> complete pooling (1 common intercept); 
# priors = ~ 0 + factor(cafe) -> no pooling (0 common intercept); 
# priors = ~ 1 + factor(cafe) -> partial pooling (1 common intercept + 1 intercept per cafe): allows shrinkage
# LKJ priors accounts for correlation coefficient; (1 + days || subject) fixes correlation between intercept and days to 0 

prior0 <- prior(normal(0, 10), class = Intercept)
m0 <- brm(
  survival ~ 1,
  family = binomial(link = "logit"),
  prior = prior0,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.8)) # adjusting the delta step size (default=.8) for better samples (slower computing)


prior1 <- c(
  prior(normal(0, 10), class = Intercept),
  prior(normal(0, 10), class = b))

m1 <- brm(
  survival ~ .,
  family = binomial(link = "logit"),
  prior = prior1,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
control = list(adapt_delta = 0.8)) #  # using the dot = "all predictors"

m2 <- brm(
  survival ~ 1 + pclass + gender + age,
  family = binomial(link = "logit"),
  prior = prior1,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.8))

m3 <- brm(
  survival ~ 1 + pclass + gender + pclass:gender + age,
  family = binomial(link = "logit"),
  prior = prior1,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.8))



## Interlude: get model prior and posterior distributions

mod <- m3 # Chose the model you want to assess
prior <- prior_samples(mod)
post <- posterior_samples(mod)
head(prior)
head(post)



## II. Prior predictive checking

prior_summary(mod)

# Prior distribution: One level
# extracts prior samples & applies the inverse link function
prior_samples(mod) %>%
  mutate(p = brms::inv_logit_scaled(Intercept) ) %>%
  ggplot(aes(x = p) ) +
  geom_density(fill = "steelblue", adjust = 0.1) +
  labs(x = "Probability a priori to survive", y = "Probability density")

# Prior distribution: Multi-level
prior_samples(mod) %>%
  mutate(
    condition1 = plogis(Intercept - 0.5 * b),
    condition2 = plogis(Intercept + 0.5 * b)
  ) %>%
  ggplot(aes(x = condition2 - condition1) ) +
  geom_density(fill = "steelblue", adjust = 0.1) +
  labs(
    x = "Difference in survival probability between conditions",
    y = "Probability density"  )


# Prior samples distribution
prior %>% 
  sample_n(size = 1e2) %>% 
  rownames_to_column("draw") %>% 
  expand(nesting(draw, Intercept, b), a = c(-2, 2) ) %>%
  mutate(d = Intercept + b * a) %>% 
  ggplot(aes(x = a, y = d)) +
  geom_line(aes(group = draw), color = "steelblue", size = 0.5, alpha = 0.5) +
  labs(x = "weight", y = "height")
# Visualize n predictions estimated from prior distribution



## III. Convergence diagnostics (if fail: go back to I. / modify fitting parameters. Check Prior distributions @end)
mod %>%
  plot(
    pars = "^b_",
    combo = c("dens_overlay", "trace"), widths = c(1, 1.5),
    theme = theme_bw(base_size = 14, base_family = "Open Sans")  )



## IV. Posterior predictive checking (if fail: go back to I. or add data. Check Prior distributions @end)
# Graphical posterior predictive checks: https://mc-stan.org/rstanarm/reference/pp_check.stanreg.html

pp_check(mod, nsamples = 1e2)
posterior_summary(mod, pars = c("^b_", "sigma"), probs = c(0.025, 0.975) )



## V. Model comparisons

m1 <- add_criterion(m1, "waic")
m2 <- add_criterion(m2, "waic")
m3 <- add_criterion(m3, "waic")

# 1. WAIC with Leave-One-Out
model_comparison_table <- loo_compare(m1, m2, m3, criterion = "waic") %>%
  data.frame %>%
  rownames_to_column(var = "model")

# 2. Akaike weights
weights <- data.frame(weight = model_weights(m1, m2, m3, weights = "waic") ) %>%
  round(digits = 3) %>%
  rownames_to_column(var = "model")
# weights gives which model generates distributions that fits the best data given other models: Akaike weight

# 3. Bayes factor -> Better use bayes_factor (in-factor) to compare hypotheses about models
bayes_R2(mod)
hist(bayes_R2(mod, summary = FALSE), showMode = TRUE, xlab = expression(rho))
plotPost(bayes_R2(mod, summary = FALSE) - bayes_R2(m2, summary = FALSE) ) # compare R2 between m3 and m2

# 4. Summary
left_join(model_comparison_table, weights, by = "model")
summary(mod) # marginal means in odds ratio

dev.m1 <- mean(-2 * rowSums(log_lik(m1) ) )
dev.m2 <- mean(-2 * rowSums(log_lik(m2) ) )
dev.m3 <- mean(-2 * rowSums(log_lik(m3) ) )
dev.m4 <- mean(-2 * rowSums(log_lik(m4) ) )
deviances <- c(dev.m1, dev.m2, dev.m3, dev.m4)
comparison <- model_comparison_table %>% data.frame %>% select(waic) %>% rownames_to_column()
waics <- comparison %>% arrange(rowname) %>% pull(waic)

# 5. Interpreting relative effect
fixed_effects <- fixef(mod) # fixed effects extraction
rethinking::logistic(fixed_effects) # == plogis(fixed_effects)
# Proportion of change on the odds induced by predictor
hist(exp(post$b_age), compVal = 1, xlab = "Odds ratio") 
plotPost(exp(post$b_age), compVal = 1, xlab = "Odds ratio") # similar

# 6. Interpreting absolute effect
intercept_samples <- plogis(post$b_age) 
# Effective impact on probabilities when predictor changes by 1 unit
hist(exp(intercept_samples), compVal = 0.5, xlab = "Effect of age on probability of surviving") 
plotPost(intercept_samples, compVal = 0.5, xlab = "Effect of age on probability of surviving") 



# Optional: R2 for parameter (for y~b_param+sigma)
beta <- post$b_mother
sigma <- post$sigma
f1 <- beta^2 * var(d1$height)
rho <- f1 / (f1 + sigma^2)
hist(rho, showMode = TRUE, xlab = expression(rho))



## VI. Visualize posterior predictions

plot(conditional_effects(mod), effects = "pclass:gender")
# Conditional_effects given the model

intercept_samples <- plogis(post$b_Intercept)
hist(intercept_samples, compVal = 0.5, xlab = "Probability of pulling left")









## Tools: Prior and Posterior checking

# PRIOR
# Visualize distribution of prior parameters
H.scv <- Hscv(x = prior, verbose = TRUE)
fhat_prior <- kde(x = prior, H = H.scv, compute.cont = TRUE)

plot(
  fhat_prior, display = "persp", col = "steelblue", border = NA,
  xlab = "\nmu", ylab = "\nsigma", zlab = "\n\np(mu, sigma)",
  shade = 0.8, phi = 30, ticktype = "detailed",
  cex.lab = 1.2, family = "Helvetica")

# Prior on intercept (normal(0, 10)
data.frame(value1 = rnorm(1e4, 0, 10) ) %>% # 10.000 samples from Normal(70, 10)
  ggplot(aes(value1) ) +
  geom_histogram(col = "white")

# Prior on beta (normal(0, 10)
data.frame(value2 = rnorm(1e4, 0, 10) ) %>% 
  ggplot(aes(value2) ) +
  geom_histogram(col = "white")






# POSTERIOR
# Visualize distribution of posterior parameters
H.scv <- Hscv(post[, 1:2])
fhat_post <- kde(x = post[, 1:2], H = H.scv, compute.cont = TRUE)

plot(fhat_post, display = "persp", col = "purple", border = NA,
     xlab = "\nmu", ylab = "\nsigma", zlab = "\np(mu, sigma)",
     shade = 0.8, phi = 30, ticktype = "detailed",
     cex.lab = 1.2, family = "Helvetica")

# Posterior samples check (with Autocorrelation for each parameter)
post <- posterior_samples(mod, add_chain = TRUE)
post %>% mcmc_acf(pars = vars(b_Intercept:b_age), lags = 10)

# Multicolinearity check (pain in the eyes; to check along with MCMC chains convergence)
pairs(mod)






# MODEL
# Represent model predictions (specifically for 1 variable)
age.seq <- data.frame(age = seq(from = min(d1$age), to = max(d1$age), 
                                      length.out = 1e2) ) # create a vector of all possible values for y

mu <- data.frame(
  fitted(mod, newdata = age.seq, probs = c(0.025, 0.975)) ) %>% 
  bind_cols(age.seq) # we get the model predictions for those values: we obtain mu

pred_height <- data.frame(
  predict(mod, newdata = age.seq, probs = c(0.025, 0.975) )) %>%
  bind_cols(age.seq) # we obtain sigma for the model

d1 %>%
  ggplot(aes(x = age, y = height) ) +
  geom_point(colour = "white", fill = "black", pch = 21, size = 3, alpha = 0.8) +
  geom_ribbon(
    data = pred_height, aes(x = age, ymin = Q2.5, ymax = Q97.5),
    alpha = 0.2, inherit.aes = FALSE  ) +
  geom_smooth(
    data = mu, aes(y = Estimate, ymin = Q2.5, ymax = Q97.5),
    stat = "identity", color = "black", alpha = 0.8, size = 1  )



# if variables are categorical: absolute effect for all parameters
model_predictions <- fitted(mod) %>%
  data.frame() %>% 
  bind_cols(d1) %>%
  mutate(gender = factor(gender), pclass = factor(pclass) )



# Model averaging 
new_data <- data.frame(
  age = seq(from = min(d1$age), to = max(d1$age), length.out = 30),
  mass = 4.5) # grid of values for which we will generate predictions

averaged_predictions <- pp_average(
  m1, m2, m3,
  weights = "waic",
  method  = "fitted",
  newdata = new_data
) %>%
  as.data.frame() %>%
  bind_cols(new_data) # predictions averaged on 4 models






# PARAMETER
# Calculate Highest Density Interval (HDI) for the most probable values of the parameter + ROPE

# Example with distribution of 1 parameter taken from samples taken from the posterior
samples <- sample(post, size = 1e3, replace = TRUE) # if need to: sample the posterior distribution

set.seed(666)
p_grid <- seq(from = 0, to = 1, length.out = 1e3) # check that [From To] values are correct
pParam <- dbeta(p_grid, 3, 10) # change parameter according to needs (check that code is working)
massVec <- pParam / sum(pParam)
samples <- sample(p_grid, size = 1e4, replace = TRUE, prob = pParam)

hist(samples, credMass = 0.89, cex = 1.5, xlab = expression(param), xlim = c(0, 1) )

# Visualize ROPE 
res = rope(samples, range = "default", ci = 0.95, ci_method = "HDI", verbose = TRUE)
plot(res) # Ugly as ****


plotPost(
  samples, cex = 2, cex.axis = 1.5, cex.lab = 2,
  xlab = expression(param),
  ROPE = c(res$ROPE_low, res$ROPE_high), compVal = 0)






# HYPOTHESIS TEST
# Difference in survival probability between male and female 

# SOLUTION 1
d1$male <- ifelse(d1$gender == "male", 1, 0) # Recode gender as dummy variable

m4 <- brm(
  survival ~ 1 + pclass + male + age,
  family = binomial(link = "logit"),
  prior = prior1,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores())

# 1. Chose the parameter you want to test
(hyp1 <- hypothesis(m4, "male = 0") )

# 2. Plot posterior vs prior
plot(hyp1, plot = FALSE, theme = theme_bw(base_size = 20, base_family = "Open Sans") )[[1]] +
  geom_vline(xintercept = 0, linetype = 2) +
  coord_cartesian(xlim = c(-10, 10) )

# 3. Compare prior and posterior for that parameter
data.frame(prior = hyp1$prior_samples$H1, posterior = hyp1$samples$H1) %>%
  gather(type, value) %>%
  mutate(type = factor(type, levels = c("prior", "posterior") ) ) %>%
  ggplot(aes(x = value) ) +
  geom_histogram(bins = 50, alpha = 0.8, col = "white", fill = "steelblue") +
  geom_vline(xintercept = 0, lty = 2, size = 1) +
  facet_wrap(~type, scales = "free") +
  labs(x = expression(beta[male]), y = "Nombre d'échantillons")

# 4. Histogram for probability of surviving in males compared to females
post <- posterior_samples(m4)
p.surviv.male <- plogis(post$b_Intercept + post$b_male)
p.surviv.female <- plogis(post$b_Intercept)
diff.surviv <- p.surviv.male - p.surviv.female
hist(diff.surviv)
plotPost(diff.surviv, compVal = 0)
mean(diff.surviv) # .20 = difference in probability of survival between males and females (females p > males p)


# SOLUTION 2: [bayestestr bayes_factor: HDI+ROPE ; point_estimate (savage-dickey) + plots]

# Point estimate (Savage-Dickey)
BF_param = bayesfactor_parameters(mod, null = 0) # compares prior and posterior samples at 1 point
plot(BF_param)

BFrope_parram=bf_rope(mod)
BFrope_parram # reads as any BF, must be >1 
plot(BF_param)


# SOLUTION 3: compare model 1 with effect of interest against model 2 without effect of interest






## Multi-levels example
setwd("C:/Users/vguigon/Dropbox (Personnelle)/Cursus/Divers/Mooc/Bayes - IMSB2021")
data <- read.csv("Cours09/data/absenteeism.csv")

prior1 <- c(
  prior(normal(0, 1), class = Intercept, coef = ""),
  prior(normal(0, 1), class = b),
  prior(cauchy(0, 1), class = sd),
  prior(lkj(2), class = cor)
)
mod1 <- brm(
  presence | trials(total) ~ 1 + reminder + (1 + reminder | researcher), 
  family = binomial(link = "logit"),
  prior = prior1,
  data = data,
  sample_prior = TRUE,
  warmup = 2000, iter = 1e4,
  chains = 4, cores = parallel::detectCores(),
  control = list(adapt_delta = 0.95))



# Other example of multi-level

d4 <- get(data(UCBadmit) )

d4 %>%
  ggplot(aes(x = dept, y = admit / applications) ) +
  geom_bar(stat = "identity") +
  facet_wrap(~ applicant.gender) +
  labs(x = "Département", y = "Probabilité d'admission")

d4$gender <- ifelse(d4$applicant.gender == "female", -0.5, 0.5)
d4$dept_id <- coerce_index(d4$dept) # create an index for department

p3 <- c(
  prior(normal(0, 10), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(cauchy(0, 2), class = "sd"),
  prior(lkj(2), class = "cor"))

m3 <- brm(
  admit | trials(applications) ~ 1 + gender + (1 + gender | dept_id),
  data = d4, family = binomial,
  prior = p3,
  warmup = 1000, iter = 5000,
  control = list(adapt_delta = 0.99, max_treedepth = 12),
  backend = "cmdstanr")

d4 %>%
  group_by(dept_id, applications) %>%
  data_grid(gender = seq_range(gender, n = 1e2) ) %>%
  add_fitted_samples(m3, newdata = ., n = 100, scale = "linear") %>%
  mutate(estimate = plogis(estimate) ) %>%
  ggplot(aes(x = gender, y = estimate, group = .iteration) ) +
  geom_hline(yintercept = 0.5, lty = 2) +
  geom_line(aes(y = estimate, group = .iteration), size = 0.5, alpha = 0.2) +
  facet_wrap(~dept_id, nrow = 2)








### Predict continuous y with categorical x ###
## Categorical predictors

# create dummy variables, e.g.,
df$clade.NWM <- ifelse(df$clade == "New World Monkey", 1, 0)
df$clade.OWM <- ifelse(df$clade == "Old World Monkey", 1, 0)
df$clade.S <- ifelse(df$clade == "Strepsirrhine", 1, 0)

# model
mod1 <- brm(
  kcal.per.g ~ 1 + clade.NWM + clade.OWM + clade.S,
  prior(normal(0.6, 10), class = Intercept),
  prior(normal(0, 1), class = b),
  family = binomial,
  data = df) # model is: ?(i) = intercept + b*categ(i)

summary(mod1)
post <- posterior_samples(mod1)

# retrieves posterior samples for each category (k-1 dummy variables)
mu.ape <- post$b_Intercept # ape will be our intercept
mu.NWM <- post$b_Intercept + post$b_clade.NWM
mu.OWM <- post$b_Intercept + post$b_clade.OWM
mu.S <- post$b_Intercept + post$b_clade.S
precis(data.frame(mu.ape, mu.NWM, mu.OWM, mu.S), prob = 0.95) # displays a summary of the posterior samples

# difference between 2 groups
diff.NWM.OWM <- mu.NWM - mu.OWM 
quantile(diff.NWM.OWM, probs = c(0.025, 0.5, 0.975) )
plotPost(diff.NWM.OWM, compVal = 0, ROPE = c(-0.1, 0.1) )


mod2 <- brm(
  kcal.per.g ~ 1 + clade.NWM + clade.OWM + clade.S,
  prior(normal(0.6, 10), class = b),
  family = binomial,
  data = df) # model is: ?(i) = intercept(categ[i]) = no intercept but a categorical predictor

summary(mod2) # get estimate for each categ