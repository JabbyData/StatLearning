library("ISLR2")
library("glmnet")
library("dplyr")
library("ggplot2")
attach(mtcars)
?mtcars
head(mtcars)
Mtcars <- data.matrix(mtcars[, -which(names(mtcars) == "mpg")])
# Ridge Regression
model1 <- cv.glmnet(Mtcars, mpg, alpha = 0,standardize = TRUE) # alpha = 0 for ridge regression, nfolds = 10 by default
# sd(mpg) # 6.026948, to check that the data has not been permanently standardized
# Standardize the data : transform quantitiave data with a mean of 0 and a standard deviation of 1 and qualitative variables into dummy variables

# Qualitative variables
# Only dummy variables
if (!is.factor(Mtcars)) {
  Mtcars <- as.factor(vs)
}

if (!is.factor(am)) {
  am <- as.factor(am)
}

# Quantitative variables
mtcars <- mtcars %>% mutate_at(c('mpg', 'disp', 'hp', 'drat', 'wt', 'qsec'), ~(scale(.) %>% as.vector))

model2 <- cv.glmnet(Mtcars,mpg,alpha=0,standardize=FALSE) # alpha = 0 for ridge regression, using glmnet's standardization

# Manual standardization of possible qualitative variables
if (!is.factor(gear)) {
  gear <- as.factor(gear)
}

if (!is.factor(carb)) {
  carb <- as.factor(carb)
}

if (!is.factor(cyl)) {
  cyl <- as.factor(cyl)
}

model3 <- cv.glmnet(Mtcars, mpg, alpha = 0,standardize = FALSE) # alpha = 0 for ridge regression, using our own standardization


# Qunatitative variables v2
mtcars <- mtcars %>% mutate_at(c('mpg', 'disp', 'hp', 'drat', 'wt', 'qsec', 'cyl', 'gear', 'carb'), ~(scale(.) %>% as.vector))
model4 <- cv.glmnet(Mtcars, mpg, alpha = 0,standardize = FALSE) # alpha = 0 for ridge regression, using our own standardization

# Plot the results
par(mfrow = c(1, 1))
plot(model1,main="glmnet's basic standardization")
plot(model2,main="manual standardization (basic qualitative variables)")
plot(model3,main="manual standardization (all qualitative variables)")
plot(model4,main="manual standardization (all qualitative variables v2)")

model1$lambda.min
model2$lambda.min
model3$lambda.min
model4$lambda.min

# Model comparison
coef1 <- coef(model1, s = model1$lambda.min)
coef2 <- coef(model2, s = model2$lambda.min)
coef3 <- coef(model3, s = model3$lambda.min)
coef4 <- coef(model4, s = model4$lambda.min)
coef4

p <- length(coef1)-1

data1 <- data.frame(variable = 1:p, coef = coef1[-1], model = "Model1")
data2 <- data.frame(variable = 1:p, coef = coef2[-1], model = "Model2")
data3 <- data.frame(variable = 1:p, coef = coef3[-1], model = "Model3")
data4 <- data.frame(variable = 1:p, coef = coef4[-1], model = "Model4")

all_data <- rbind(data1,data2,data3,data4)

ggplot(all_data, aes(x = variable, y = coef, color = model)) +
  geom_point(size = 3) +
  labs(x = "Variable", y = "Coefficient", color = "Model") +
  theme_minimal()


# plot coefficents of the model 1 as a function of lambda
# create a vector of lambda values from 0.5 to 2^8 in log scale
lambdas <- exp(seq(from = log(0.5), to = log(50), length.out = 100))
# compute the coefficients for each lambda
coefs <- coef(model1, s = lambdas)
# vector containing names of predictors in coefs, excluding the intercept
predictors <- rownames(coefs)[-1]
p <- length(predictors)

# plot the coefficients as a function of lambda for model1 (R ridge regression)
plot(x = lambdas, y = coefs[2,], type = "l", xlab = "log(lambda)", ylab = "coefficient", col = 2, ylim=c(-0.15,0.4))
for (i in 3:p) {
  lines(x = lambdas, y = coefs[i,],type="l", col = i)
}
legend("topright", legend = predictors, col = 1:p, lty = 1, cex = 0.8)


# LASSO regression
model5 <- cv.glmnet(Mtcars, mpg, alpha = 1,standardize = TRUE) # alpha = 1 for lasso regression, nfolds = 10 by default
plot(model5,main="glmnet's basic standardization")
model5$lambda.min
lambdas = exp(seq(from = log(0.5), to = log(8), length.out = 100))
coef5 <- coef(model5, s = lambdas)
predictors <- rownames(coef5)[-1]
p <- length(predictors)
max(coef5)
min(coef5)

# plot coefficients as function of lambda for model5 (LASSO regression)
plot(x = lambdas, y = coef5[2,], type = "l", xlab = "log(lambda)", ylab = "coefficient", col = 2, ylim=c(-3,1))
for (i in 3:p) {
  lines(x = lambdas, y = coef5[i,],type="l", col = i)
}

legend("topright", legend = predictors, col = 1:p, lty = 1, cex = 0.8)

################ HUGE PROBLEM : data standardized ############################
