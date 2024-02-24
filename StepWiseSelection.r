library("ISLR2")
library("boot")
attach(Credit)
?Credit
#head(Credit)

# Linear regression : predict balance

# Preprocess data
# Analyse of qualitative variables

#unique(Credit$Region) # number of possible values in the region variable
#contrasts(Region) # coding of the qualitative variable
#contrasts(Own)

# Making sure that the qualitative variables are factors (cf working with set of values instead of characters)
if (!is.factor(Region)) {
  Region <- as.factor(Region)
}
if (!is.factor(Student)) {
  CStudent <- as.factor(Student)
}
if (!is.factor(Married)) {
  Married <- as.factor(Married)
}
if (!is.factor(Own)) {
    Own <- as.factor(Own)
}

n <- dim(Credit)[1]

# Best model selection using StepWise Method
MSEs <- rep(0,10)
Preds <- list()
# Starting with the null model
#lm.fit <- lm(Balance~1,data=Credit) # predict balance's mean, not using it because hide other results
all.equal(summary(lm.fit)$coef[1], mean(Balance)) # check equality between the two values
preds_available <- 1:10
# preds_available <- preds_available[-which(preds_available == 2)] # drop the elt equals to 2

for (p in 1:10){
  cat("step : ",p,"\n")
  MSE_min <- 1000000
  for (pred in preds_available){
    lm.fit <- glm(Balance~.,data=Credit[,c(pred,11)])
    MSE <- cv.glm(Credit,lm.fit,K=8)$delta[1]
    if (MSE < MSE_min){
      Pred <- pred
      MSE_min <- MSE
    }
  }
  MSEs[p] <- MSE_min
  Preds <- append(Pred,Preds)
  preds_available <- preds_available[-which(preds_available==Pred)]
}

# plot results
plot(1:10,MSEs,xlab="nb of predictors",ylab="MSE CV",type="b",col="red")

# displaying best predictors
for (k in 1:length(Preds)){
  preds <- unlist(Preds[1:k])
  cat(k,":",names(Credit)[preds],"\n")
}