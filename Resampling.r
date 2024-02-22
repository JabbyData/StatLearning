library("ISLR2")
library("boot")
attach(Auto)

# dataset obersvation
?Auto
head(Auto)
pairs(Auto)

# Linear regression Auto ~ Horsepower using different polynomial degrees
# and different splits of the dataset

# 10 different datasets
dim(Auto)[1] # number of observations
sep <- dim(Auto)[1]%/%10 # int division 392/10

MSE_test <- NULL
R2_test <- NULL

for (i in 0:9){
  # validation subset between indices sep * i + 1 and sep * (i+1)
  valid <- 1:dim(Auto)[1] >= sep * i + 1 & 1:dim(Auto)[1] <= sep * (i+1)
  # linear regression for 9 different polynomial degrees
  for (k in 1:9){
    # training
    lr.fit <- lm(mpg~horsepower + I(horsepower^k),data=Auto[!valid,])
    # validation
    MSE_test <- c(MSE_test,mean((Auto[valid,1] - predict.lm(lr.fit,Auto[valid,]))^2))
    R2_test <- c(R2_test,1 - sum((Auto[valid,1] - predict.lm(lr.fit,Auto[valid,]))^2)/sum((Auto[valid,1] - mean(Auto[valid,1]))^2))
  }
}
# dividing figures
par(mfrow=c(2,1))

# plot the MSE for each degree on various datasets
plot(1:9,MSE_test[1:9],xlab="degree",ylab="MSE",col=0,type="b",ylim=c(9,28),main="mpg ~ hp^degree validation MSE for different datasets")
for (i in 1:9){
  start <- 9*i+1
  end <- 9*(i+1)
  lines(1:9,MSE_test[start:end],xlab="degree",ylab="MSE",col=i,type="b",ylim=c(9,28))
}

# plot the R2 for each degree on various datasets
plot(1:9,R2_test[1:9],xlab="degree",ylab="R2",col=0,type="b",ylim=c(0,1),main="mpg ~ hp^degree validation R2 for different datasets")
for (i in 1:9){
  start <- 9*i+1
  end <- 9*(i+1)
  lines(1:9,R2_test[start:end],xlab="degree",ylab="R2",col=i,type="b",ylim=c(0,1))
}

# As expected, R2 and MSE are optimal for k = 2

# K-fold cross validation for k = 2
MSE_opti <- NULL
for (i in 0:9){
  MSE_opti <- c(MSE_opti,MSE_test[9*i+2])
}

MSE_opti <- mean(MSE_opti)
MSE_opti # around 21

# K-fold cross validation for k = 2 using library boot
set.seed(1) # for reproductibility
glm.fit <- glm(mpg~horsepower + I(horsepower^2),data=Auto)
CV_MSE <- rep(0,14)
# K-fold cross validation for different values of K
for (k in 2:15){
  CV_MSE[k-1] <- cv.glm(Auto,glm.fit,K=k)$delta[1]
}

plot(2:15,CV_MSE,xlab="K",ylab="MSE",type="b",main="K-fold cross validation for different values of K")


# Bootstrapping
alpha.fn <- function(data, index){
  X <- data$X[index]
  Y <- data$Y[index]
  (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y))
}

alpha.fn(Portfolio , 1:100) # Estimates of alpha using the first 100 observations

# With remplacement
set.seed(1) # for reproductibility
alpha.fn(Portfolio , sample(100,100,replace=TRUE)) # around 0.576

# Using boot
boot(Portfolio,alpha.fn,R=1000) # 1000 bootstrap estimates for alpha, we find an estimate around 0.576

# Linear Regression
boot.fn <- function(data , index){
  coef(lm(mpg ~ horsepower , data = data , subset = index))
}
boot(Auto,boot.fn,R=1000) # sampling estimation of slope and intercept for the LR mpg ~ horsepower
summary(lm(mpg ~ horsepower , data = Auto)) # Results are very close