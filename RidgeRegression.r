library("ISLR2")
library("glmnet")
library("corrplot")
?Hitters

# Remove missing values
Hitters <- na.omit(Hitters)

# Preprocess data
MHitters <-model.matrix(Salary~.,Hitters)[,-1] # Design matrix to prepare prediction of Salary
head(MHitters)
YHitters <- Hitters$Salary # Response variable

# Simple Ridge regression
grid <- 10^seq(10,-2,length=100) # creating lambda from 10^10 to 10^-2, always in that order
ridge.mod1 <- glmnet(MHitters,YHitters,alpha=0,lambda=grid) # alpha=0 for ridge regression

dim(coef(ridge.mod1)) # 20 coefficients for 100 lambda values
coef1 <- coef(ridge.mod1)

# PLotting coefficients as function of lambda
grid <- log(grid,base=10)
plot(grid,coef1[2,],xlab="Log Lambda",ylab="Coefficient Estismate",type='l',ylim=c(-20,20),col=2,main="LASSO Regression to predict Salary")
for (p in 3:dim(coef1)[1]){
  lines(grid,coef1[p,],type='l',col=p)
}
predictors <- rownames(coef1)[-1]
legend("topright",legend=predictors,col=1:19,lty=1,cex=0.8)

# Manual Ridge Regression
#set.seed(1)
train <- sample(1:nrow(MHitters), 2 * nrow(MHitters)/3) # select randomly 2/3 of the valid data for training
grid <- 10^grid # 10^10 -> 10^-2
ridge.mod <- cv.glmnet(MHitters[train,],YHitters[train],alpha=0,lambda=grid)
ridge.pred <- predict(ridge.mod,s=4,newx=MHitters[-train,])
mean((ridge.pred - YHitters[-train])^2)

# Cross Validation Ridge Regression
cv.out <- cv.glmnet(MHitters[train,],YHitters[train],alpha=0) # library performs the cross validation for multiple value of lambda
plot(cv.out)
cv.out$lambda.min # value that minimizes the MSE

# Comparing results
cv.out.pred <- predict(cv.out,s=cv.out$lambda.min,newx = MHitters[-train,])
MSE_CV <- mean((cv.out.pred - YHitters[-train])^2)

ridge.pred <- predict(ridge.mod,s=cv.out$lambda.min,newx = MHitters[-train,])
MSE_S <- mean((ridge.pred - YHitters[-train])^2)

# MSE comparison
MSE_S
MSE_CV

# training on the full model
out <- glmnet(MHitters,YHitters,alpha=0)
predict(out,type="coefficients",s=cv.out$lambda.min) # no variable selection (no coeff equals zero)

# LASSO regression
lasso.mod <- cv.glmnet(MHitters[train,],YHitters[train],alpha=1)
plot(lasso.mod)
lasso.mod$lambda.min

# MSE
lasso.pred <- predict(lasso.mod,s=lasso.mod$lambda.min,newx=MHitters[-train,])
MSE_LCV <- mean((lasso.pred - YHitters[-train])^2)
MSE_LCV

# verif on the whole dataset
out <- glmnet(MHitters,YHitters,alpha=1,lambda=grid)
predict(out,s=lasso.mod$lambda.min,type="coefficients")


# Obj : find optimal alpha (because numerous variables are correlated)
corrplot(cor(MHitters))

alphas <- seq(0,1,length=100)
MSEs <- NULL
MSE_min <- 10000000
alpha_min <- -1
for (alpha in alphas){
  cat("alpha : ",alpha,"\n")
  # Elastic Net Model
  lasso.mod <- glmnet(MHitters[train,],YHitters[train],alpha=alpha,lambda = grid) # initial model
  EN.model <- cv.glmnet(MHitters[train,],YHitters[train],alpha=alpha) # 10-folds CV to find the optimal alpha
  lasso.pred <- predict(lasso.mod,s=lasso.mod$lambda.min,newx=MHitters[-train,])
  # MSE
  MSE_CV <- mean((lasso.pred - YHitters[-train])^2)
  if (MSE_CV < MSE_min){
    MSE_min <- MSE_CV
    alpha_min <- alpha
  }
  MSEs <- c(MSEs,MSE_CV)
}

# MSE as function of alpha
plot(alphas,MSEs,type="l",main="MSE for optimal alpha found using CV") # seems that Ridge Regression is the best model to choose.


# Comparison with a simple linear regression
lm.fit <- lm(Salary~.,data=Hitters[train,])
lm.pred <- predict(lm.fit,newdata = Hitters[-train,])
MSE_LM <- mean((lm.pred - YHitters[-train])^2)

MSE_LM
MSE_min

# Linear model seems to be still more efficient than ridge regression