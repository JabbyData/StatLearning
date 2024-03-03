library(pls)
library(ISLR2)
set.seed(2) # for reprocductibility

attach(Hitters)
Hitters <- na.omit(Hitters) # removing missing values

# Simple PCR
pcr.fit <- pcr(Salary~.,data=Hitters,scale=TRUE,validation="CV")
summary(pcr.fit) # reports the root mean squared error (RMSE)
validationplot(pcr.fit,val.type="MSE") # performances

# Train Test PCR
train <- sample(1:nrow(Hitters), 2 * nrow(Hitters)/3)
# Selection of the best nb of comp using CV
pcr.fit <- pcr(Salary~.,data=Hitters,subset=train,scale=TRUE,validation="CV")
summary(pcr.fit)
validationplot(pcr.fit,val.type="MSE") # ncomp = 5 seems to minimizes the training MSE

pcr.pred <- predict(pcr.fit,newdata = Hitters[-train,],ncomp=5)
names(Hitters)
mean((pcr.pred - Hitters[-train,19])^2) # interesting MSE compared to ridge regression models

# Performing PCR on the whole dataset
MHitters <- model.matrix(Salary~.,Hitters)[,-1] # preprocessed model for LR
YHitters <- Hitters[,19]
pcr.fit <- pcr(YHitters~MHitters,scale=TRUE,ncomp=5)
summary(pcr.fit)

# Linear regression
lm.fit <- lm(Salary~.,data=Hitters,subset=train)
lm.pred <- predict(lm.fit,newdata = Hitters[-train,])
mean((lm.pred - Hitters[-train,19])^2) # MSE for the linear regression model, stil inferior to PCR

# Partial Least Squares Regression
pls.fit <- plsr(Salary~.,data=Hitters,subset=train,scale=TRUE,validation="CV")
summary(pls.fit)
validationplot(pls.fit,val.type="MSE")

pls.pred <- predict(pls.fit,newdata=MHitters[-train,],ncomp=9)
mean((pls.pred - Hitters[-train,19])^2) # more interesting

# Implementation on the whole dataset
pls.fit <- plsr(Salary~.,data=Hitters,scale=TRUE,ncomp=9)
summary(pls.fit)