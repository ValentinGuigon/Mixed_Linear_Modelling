##### Notes on lavaan and SEM #####

# Endogenous variables are hypothesized variables caused by other variables in the model. 
# These are usually the dependent variables in the model while exogenous variables are not influenced by other variables 
# in the model but instead, influence endogenous variables. 
# Endogenous variables usually have arrows pointing to them while exogenous variables don’t.

# Terrence Jorgensen: "You don't have to center anything [...] lavaan will automatically partition its variance into the level-specific components."
# link: https://groups.google.com/g/lavaan/c/h2GiNsxBIdk

##### VARIABLES #####

# Free memory
rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memory and report the memory usage
set.seed(12345)

list.of.packages <- c("rprojroot", "lavaan", "semTools")
{
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  lapply(list.of.packages, require, character.only = TRUE)
}


# Choose if dataset test or task dataset
dataset_test = 1

# Build a dataframe where X1, X2, MODERATOR are continuous and MEDIATOR, Y are binomial

if (dataset_test == 1) { # If need test data
  
  data <- as.data.frame(mvrnorm(5e2, rep(0, 5), matrix(.25, 5, 5) + .75 * diag(5)))
  colnames(dataset_test) <- c("X1", "X2", "MEDIATOR", "MODERATOR", "Y")
  data$X2 <- dat$X1 ^ 2
  data$MEDIATOR <- rbinom(nrow(data), 1, plogis(data$MEDIATOR))
  data$Y <- rbinom(nrow(data), 1, plogis(data$Y))
  head(data)
  
} else if (dataset_test == 0) { # If data already present, input your data here
}

data = data[,c("X1", "X2", "MODERATOR", "MEDIATOR", "Y")]


##### CASE 1: surface interactions model with CONTINUOUS moderator #####

model_simple = "
MEDIATOR ~ a11 * X1 + a12 * X2 + w10 * MODERATOR + w11 * X1:MODERATOR + w12 * X2:MODERATOR
  Y ~ c1 * X1 + c2 * X2 + w30 * MODERATOR + w31 * X1:MODERATOR + w32 * X2:MODERATOR + b1 * MEDIATOR
  a11b1 := a11 * b1
  a12b1 := a12 * b1
           "

# Fit the simple mediation model
fit_simple_mediation <- lavaan::sem(model_simple, data = data, se = "bootstrap", bootstrap = 100)

# DisplaY the results
summary(fit_simple_mediation, standardized = TRUE)

# Plot the mediation path diagram
lavaanPlot::lavaanPlot(model = fit_simple_mediation, coefs=T)


##### CASE 2: thorough interactions model with CONTINUOUS moderator #####

# The model is written with less specification (i.e., no  a11b1 nor a12b1) but the end result regarding the computations is the same

# Define model
model_complex = 
  "
Y ~ cprime1*X2 + cprime2*X1 + w1prime1*X2:MODERATOR + w1prime2*X1:MODERATOR + mprime*MEDIATOR + w1prime*MODERATOR + w1prime3*MEDIATOR:MODERATOR
MEDIATOR ~ a1*X2 + a2*X1 + w2prime*MODERATOR + w2prime1*X2:MODERATOR + w2prime2*X1:MODERATOR
"

# Fit the simple mediation model
fit_complex_mediation <- lavaan::sem(model_complex, data = data, se = "bootstrap", bootstrap = 100)

# Display the results
summary(fit_complex_mediation, standardized = TRUE)

# Plot the mediation path diagram
lavaanPlot::lavaanPlot(model = fit_complex_mediation, coefs=T)


##### CASE 3: multilevel model: with CONTINUOUS mediator #####

# Did not correctly fit

data_multilevel = data
data_multilevel $X2 = scale(data_multilevel $X2, center=TRUE, scale=TRUE)
data_multilevel $X1 = scale(data_multilevel $X1, center=TRUE, scale=TRUE)
data_multilevel $MEDIATOR = scale(data_multilevel $MEDIATOR, center=TRUE, scale=TRUE)


# Define model
model_multilevel = 
  "
  level: 1
  Y ~ cprime1*X2 + cprime2*X1 + w1prime1*X2:MODERATOR + w1prime2*X1:MODERATOR + mprime*MEDIATOR + w1prime*MODERATOR + w1prime3*MEDIATOR:MODERATOR
  MEDIATOR ~ a1*X2 + a2*X1 + w2prime*MODERATOR + w2prime1*X2:MODERATOR + w2prime2*X1:MODERATOR
  
  level: 2
  Y ~ yprime1*X2 + yprime2*X1 + x1prime1*X2:MODERATOR + x1prime2*X1:MODERATOR + zprime*MEDIATOR + x1prime*MODERATOR + x1prime3*MEDIATOR:MODERATOR
  MEDIATOR ~ y1*X2 + y2*X1 + z2prime*MODERATOR + x2prime1*X2:MODERATOR + x2prime2*X1:MODERATOR
"

# Fit the simple mediation model
fit_multilevel_mediation <- lavaan::sem(model_multilevel, data = data_multilevel, cluster = "Subject", se = "bootstrap", bootstrap = 100)

# Display the results
summary(fit_multilevel_mediation, standardized = TRUE)

# Plot the mediation path diagram
lavaanPlot::lavaanPlot(model = fit_multilevel_mediation, coefs=T)

# Warnings: 
# 1. Level-1 variable “Y” has no variance within some clusters.
# 2. Level-1 variable “MEDIATOR” has no variance within some clusters.
# 3. The variance-covariance matrix of the estimated parameters (vcov) does not appear to be positive definite! 
# The smallest eigenvalue (= -1.194229e+04) is smaller than zero. This may be a symptom that the model is not identified.


##### CASE 4: simple interactions model with CATEGORICAL (binomial) moderator #####

# Don't know yet how valid is the model


# Define model
model_categ_simple = 
  "
Y ~ c1*X2 + c2*X1 + m*MEDIATOR + w1*MODERATOR + wc1*X2:MODERATOR + wc2*X1:MODERATOR
MEDIATOR ~ a1*X2 + a2*X1 + w2*MODERATOR + wa1*X2:MODERATOR + wa2*X1:MODERATOR
"
# Y ~ c1*X2 + c2*X1 + m*MEDIATOR + w1*MODERATOR + wc1*X2:MODERATOR + wc2*X1:MODERATOR + wm*MEDIATOR:MODERATOR

# Fit the simple mediation model
fit_simple_categ_mediation <- lavaan::sem(model_categ_simple, data = data, estimator = "dwls", se = "bootstrap", bootstrap = 100, ordered = c("MODERATOR", "Y"))

# Display the results
summary(fit_simple_categ_mediation, standardized = TRUE)

# Plot the mediation path diagram
lavaanPlot::lavaanPlot(model = fit_simple_categ_mediation, coefs=T)


##### CASE 5: thorough interactions model with CATEGORICAL (binomial) moderator #####

## Moderated mediation: thorough interactions model
## Key parameter to model the interaction between continuous treatment and binomial moderator is:
##  ordered = c("MODERATOR", "Y")
## You can also specify: ordered = TRUE -> supposed to order all endogenous variables

# Define model
model_categ_complex = 
  "
  # total effect
  Y ~ c1*X2 + c2*X1 + m*MEDIATOR + w1*MODERATOR + wc1*X2:MODERATOR + wc2*X1:MODERATOR
  # mediation effect
  MEDIATOR ~ a1*X2 + a2*X1 + w2*MODERATOR + wa1*X2:MODERATOR + wa2*X1:MODERATOR
  # indirect and total effect, conditional on MODERATOR == 0
  ab01 := a1*m          # + 0*wa1*m
  total01 := ab01 + c1  # + 0*wc1
  ab02 := a2*m          # + 0*wa2*m
  total02 := ab02 + c2  # +0*wc2
  # indirect and total effect, conditional on MODERATOR == 1
  ab11 := ab01 + 1*wa1*m
  total11 := ab11 + c1 + 1*wc1
  ab12 := ab02 + 1*wa2*m
  total12 := ab12 + c2 + 1*wc2
"

# Fit the simple mediation model
fit_complex_categ_mediation <- lavaan::sem(model_categ_complex, data = data, estimator = "DWLS", se = "bootstrap", bootstrap = 100, ordered = c("MODERATOR", "Y"))
# ordered: Character vector. Only used if the data is in a data.frame. Treat these variables as ordered (ordinal) variables, 
#   if they are endogenous in the model. Importantly, all other variables will be treated as numeric (unless they are declared as ordered in the data.frame.) 
#   Since 0.6-4, ordered can also be logical. If TRUE, all observed endogenous variables are treated as ordered (ordinal). 
#   If FALSE, all observed endogenous variables are considered to be numeric (again, unless they are declared as ordered in the data.frame.)

# Display the results
summary(fit_complex_categ_mediation, standardized = TRUE, rsquare=T)
parameterEstimates(fit_complex_categ_mediation, standardized=T, rsquare=T)

# Plot the mediation path diagram
lavaanPlot::lavaanPlot(model = fit_complex_categ_mediation, coefs=T)
