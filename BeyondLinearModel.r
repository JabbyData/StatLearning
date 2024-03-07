library(ISLR2)
library(pscl)
library(car)
library(splines)
library(gam)
attach(Wage)
?Wage

# Analyse the data
# Obj : predict wage using age
plot(age,wage) # seems that a 2nd degree pol might be a good fit

# Model fitting
names(Wage)
Wage <- na.omit(Wage)
n <- nrow(Wage)
train <- sample(1:n,2*n/3)
lm.fit <- lm(wage~age + I(age^2),data=Wage,subset=train)
summary(lm.fit)
lm.pred <- predict(lm.fit,newdata = Wage,subset=-train)

MSE <- mean((lm.pred - Wage[,11])^2)
plot(lm.fit) # observe results

# Logistic regression on wage > 250
glm.fit <- glm(wage>250~age+I(age^2),family="binomial",data=Wage)
summary(glm.fit)
plot(glm.fit)
pR2(glm.fit)["McFadden"] # 0.03 ... bof
vif(glm.fit)

glm.preds <- predict(glm.fit,type="response")
log.odds <- log(glm.preds/(1-glm.preds))
plot(age,log.odds)
f <- function(x){
  return(-3.5*(x-20)*(x-30)/(30*20) -6*(x-50)*(x-30)/(30*10) -5*(x-50)*(x-20)/(-20*10))
}
x <- 20:80
y <- f(x)
plot(x,y) # not really the same form

# Another approach
lm.fit <- lm(wage~poly(age,4),data=Wage) # fitting using a orthogonal basis matrix for 4th degree poly
summary(lm.fit)

# performing poly reg
agelims <- range(age) # takes limits of age
age.grid <- seq(from=agelims[1],to=agelims[2]) # sequence of values within 18 and 80
lm.preds <- predict(lm.fit,newdata=list(age=age.grid),se=TRUE)
se.bands <- cbind(lm.preds$fit + 2*lm.preds$se.fit, lm.preds$fit - 2*lm.preds$se.fit) # 95% confidence interval

# plotting results
par(mfrow=c(1,2),mar=c(4.5,4.5,1,1),oma=c(0,0,4,0))
plot(age , wage , xlim = agelims , cex = .5, col = "darkgrey")
title("Degree -4 Polynomial", outer = T)
lines(age.grid , lm.preds$fit , lwd = 4, col = "blue") # plots fit curve
matlines(age.grid , se.bands , lwd = 3, col = "blue", lty = 10) # plots confidence intervals

# Determine which degree is best : ANOVA (ANalysis Of VAriance) test
fit.1 <- lm(wage ~ age , data = Wage)
fit.2 <- lm(wage ~ poly(age,2) , data = Wage)
fit.3 <- lm(wage ~ poly(age,3) , data = Wage)
fit.4 <- lm(wage ~ poly(age,4) , data = Wage)
fit.5 <- lm(wage ~ poly(age,5) , data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5) # seems that a cubic or quadratic fit is good to fit the data using polys


# logistic regression test
fit <- glm(I(wage>250)~poly(age,4),data=Wage,family=binomial) # usage of I to impose numerical logical comparison (0 or 1)
preds <- predict(fit,newdata=list(age=age.grid),se=T)

# sigmoid function to obtain p(wage>250|age)
sigmoid <- function(x){
  return(exp(x)/(1+exp(x)))
}
pfit <- sigmoid(preds$fit)
se.bands.logit <- cbind(preds$fit + 2 * preds$se.fit,preds$fit - 2 * preds$se.fit)
se.bands <- exp(se.bands.logit) / (1 + exp(se.bands.logit)) # can't apply function here...

plot(age , I(wage > 250) , xlim = agelims , type = "n", ylim = c(0, .2)) # n to ensure nothing is plotted
points(jitter(age), I((wage > 250)/5), cex = .5, pch = "|", col = "darkgrey") # /5 to ensure visibility on the graph (lim 0->0.2)
lines(age.grid , pfit , lwd = 2, col = "blue") # plotting fit curve
matlines(age.grid , se.bands , lwd = 1, col = "red", lty = 3)

# step function
table(cut(age , 4))
fit <- lm(wage ~ cut(age , 4), data = Wage)
summary(fit)

# Splines
sp.fit <- lm(wage~bs(age,knots=c(25,40,60)),data=Wage)
summary(sp.fit)
sp.pred <- predict(sp.fit,newdata=list(age=age.grid),se=T)
plot(age,wage,col="gray",main="Basic Spline")
lines(age.grid , sp.pred$fit , lwd = 2)
lines(age.grid , sp.pred$fit + 2 * sp.pred$se, lty = "dashed")
lines(age.grid , sp.pred$fit - 2 * sp.pred$se, lty = "dashed")
abline(v=25,col="red")
abline(v=40,col="red")
abline(v=60,col="red")

# Smooth Spline
plot(age,wage,col="gray",main="Smooth Spline")
sm.fit <- smooth.spline(age,wage,df=16) # choose the effective degrees of freedom (Tr(Smoother Matrix))
sm.fit2 <- smooth.spline(age,wage,cv=T) # LOOCV to find to optimal tuning parameter.
lines(sm.fit,col="red",lwd=2)
lines(sm.fit2,col="purple",lwd=2)
legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","purple"),lwd=1)

# GAMs
gam.m1 <- gam(wage~s(age,5)+education,data=Wage)
gam.m2 <- gam(wage~year+s(age,5)+education,data=Wage)
gam.m3 <- gam(wage~s(year,4)+s(age,5)+education,data=Wage)
anova(gam.m1,gam.m2,gam.m3,test="F") # anova not really adapted to GAMs