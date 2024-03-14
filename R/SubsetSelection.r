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

# best model selection
# first version manually
# Creation of training and test sets
n <- dim(Credit)[1] # n = 400, p = 11 => choose of subsets with length 50 (same length and > 11)

# Perform average MSE evaluation on each tuple of predictors
MSEs <- list() # vector to store MSEs
MSEs_R <- list() # idem but R version
Preds <- list() # vector to store associated preds
K <- 8 # number of subsets for K-fold CV
for (p in 1:10){ # number of variables
  cat("step : ",p,"\n") # display process advancement
  combi <- combn(10,p)
  MSE_min <- 1000000
  MSE_min_R <- 1000000
  tuple <- NULL
  tuple_R <- NULL
  # Linear regression with selection of the best model using average MSE
  for (i in 0:(length(combi)%/%p - 1)){
    start <- i * p + 1
    end <- (i+1) * p
    preds <- combi[start:end] # preds selection (tuple with length p)
    # Manual K-fold CV
    MSE_av <- 0
    for (k in 0:(K-1)){
      valid <- 1:n > k * 50 & 1:n <= (k+1) * 50 # validation set
      lm.fit <- lm(Balance~.,data=Credit[!valid,c(preds,11)])
      MSE_av <- MSE_av + mean((predict.lm(lm.fit,newdata=Credit[valid,]) - Credit[valid,11])^2)
    }
    MSE_av <- 1/K * MSE_av
    if (MSE_av < MSE_min){
      MSE_min <- MSE_av
      tuple <- preds
    } # R K-fold
    glm.fit <- glm(Balance~., data = Credit)
    MSE_av_R <- cv.glm(Credit,glm.fit,K=8)$delta[1]
    if (MSE_av_R < MSE_min_R){
      MSE_min_R <- MSE_av_R
    }
  }
  MSEs <- append(MSEs,MSE_min)
  MSEs_R <- append(MSEs_R,MSE_min_R)
  Preds <- append(Preds,tuple)
}

# Observe results
# Plotting MSEs versus nb of predictors

# plot MSEs
par(mfrow= c(2,1))
plot(1:10,MSEs,type="b",xlab = "nb of predictors",ylab="CV MSE",main = "Best CV MSEs versus the number of predictors used (manual)",col="red")
# lines(1:10,MSEs_R,type="b",col="blue") # to plot on the same graph
plot(1:10,MSEs_R,type="b",xlab = "nb of predictors",ylab="CV MSE_R",main = "Best CV MSEs versus the number of predictors used (R)",col="blue")

# Displaying best predictors
s <- 0
for (k in 1:10){
  start <- s+1
  end <- s+k
  cat(k,":",names(Credit)[unlist(Preds[start:end])],"\n")
  s <- s+k
}