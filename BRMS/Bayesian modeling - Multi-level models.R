library(lme4)
library(rethinking)
library(brms)

# my_packages <- c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "coda", "LearnBayes", "mcmc", "MCMCpack", "MuMIn", "reshape2", "tidybayes", "bayesplot", "shinystan", "patchwork")
# lapply(my_packages, require, character.only = TRUE)   

data(sleepstudy)
head(sleepstudy, 20)

sleepstudy %>%
  ggplot(aes(x = Days, y = Reaction) ) +
  geom_smooth(method = "lm", colour = "black") +
  geom_point() +
  facet_wrap(~Subject, nrow = 2) +
  scale_x_continuous(breaks = c(0, 2, 4, 6, 8) )

# Mod1 = l'effet fixe de Days
# Mod2 = l'effet fixe de Days + un effet aléatoire de Subject (varying intercept)
# Mod3 = l'effet fixe de Days + un effet aléatoire de Subject (varying intercept) + 
#       un effet aléatoire de Days (varying slope)

### Mod1
mod1 <- brm(
  Reaction ~ 1 + Days,
  prior = c(
    set_prior("normal(200, 20)", class = "Intercept"),
    set_prior("normal(0, 10)", class = "b"),
    set_prior("cauchy(0, 10)", class = "sigma")
  ),
  data = sleepstudy,
  warmup = 1000, iter = 5000,
  cores = parallel::detectCores()
)

summary(mod1)
plot(
  mod1, combo = c("dens_overlay", "trace"),
  theme = theme_bw(base_size = 16, base_family = "Open Sans")
)





### Mod2
mod2 <- brm(
  Reaction ~ 1 + Days + (1 | Subject),
  prior = c(
    set_prior("normal(200, 20)", class = "Intercept"),
    set_prior("normal(0, 10)", class = "b"),
    set_prior("cauchy(0, 10)", class = "sigma"),
    set_prior("cauchy(0, 10)", class = "sd")
  ),
  data = sleepstudy,
  warmup = 1000, iter = 5000,
  cores = parallel::detectCores()
)

summary(mod2)
plot(
  mod2, combo = c("dens_overlay", "trace"),
  theme = theme_bw(base_size = 16, base_family = "Open Sans")
)



### Mod3
mod3 <- brm(
  Reaction ~ 1 + Days + (1 + Days | Subject),
  prior = c(
    set_prior("normal(200, 20)", class = "Intercept"),
    set_prior("normal(0, 10)", class = "b"),
    set_prior("cauchy(0, 10)", class = "sigma"),
    set_prior("cauchy(0, 10)", class = "sd")
  ),
  data = sleepstudy,
  warmup = 1000, iter = 5000,
  cores = parallel::detectCores()
)

summary(mod3)
plot(
  mod3, combo = c("dens_overlay", "trace"),
  theme = theme_bw(base_size = 16, base_family = "Open Sans")
)



### Comparisons between models
posterior_summary(mod1, pars = c("^b", "sigma") )
posterior_summary(mod2, pars = c("^b", "sigma") )
posterior_summary(mod3, pars = c("^b", "sigma") )

mod1 <- add_criterion(mod1, "waic")
mod2 <- add_criterion(mod2, "waic")
mod3 <- add_criterion(mod3, "waic")

w <- loo_compare(mod1, mod2, mod3, criterion = "waic")
print(w, simplify = FALSE)

model_weights(mod6, mod7, mod8, weights = "waic")