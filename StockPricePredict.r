############################################
# Stock Price Prediction (on the Kaggle dataset for Tesla 2010 - 2020 )
############################################

# loading the dataset
Dataset <- read.csv("Data/StockPricePrediction_data/TSLA.csv")

# Data Preprocessing
head(Dataset)
attach(Dataset)

# Displaying prices (close and open) in the same graph to observe tendency
Date <- as.Date(Date)
plot(Date, Close, pch = 20 , col = "red", xlab = "Date", ylab = "Close Price")
lines(Date, Open, col = "green")

# Multiple Linear Regression High against ohther variables
MLRHighAll <- lm(High ~ Open + Low + Close + Volume, data = Dataset)
summary(MLRHighAll)

# Bof tim dependencies so model is very biased ... need time series to continue
