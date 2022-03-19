---
output:
  word_document: default
  html_document: default
  pdf_document: default
---
# UNIVERSITY OF TORONTO MISSISSAUGA STA314 KAGGLE PREDICTION COMPETITION 2021 
BY: AMRINDER SEHMBI 



```
# Set working directory
setwd("~/Sta314 kaggle competion")

# Read in the data
# d.train is the training set
d.train = read.csv('trainingdata.csv')
# d.test are the predictors in the test set
d.test = read.csv('test_predictors.csv')

# Load Libraries
library(glmnet)
library(gbm)
library('splines')
library('gam')

# Response variable
y = d.train$y

# Explanatory variable
x = model.matrix(d.train$y ~ . ,d.train)[,-1]

# Set random seed to 1
set.seed(1)

# Randomly sample half of given data set for training model
train = sample(1:nrow(x),nrow(x)/2)

# Separate other half of given data set for testing model
test = (-train)

# Response variable of test data set
ytest = y[test]

```


```
############################### LASSO Regression ###############################
# Create grid for lambda
grid = 10^seq(10,-2,length = 100)
#Base lasso model
lasso.mod = glmnet(x[train,],y[train],alpha =1, lambda = grid)
```


```
# Set Random Seed
set.seed(1)
```


```
# Run cross validation for lasso to choose optimal lambda value 
cv.la = cv.glmnet(x[train,],y[train],alpha =1, lambda = grid)
```


```
# Optimal Lambda value for lasso model
lambda_best <- cv.la$lambda.min
```


```
# Optimal lasso model
la = glmnet(x[train,],y[train],alpha =1, lambda = lambda_best)
```


```
# Training set prediction
predictions_train <- predict(la, s = lambda_best, newx = x[train,])
```


```
# Rooted mean squared error
sqrt(mean((predictions_train - y[train])^2))
```


```
# Plot lasso model
plot(lasso.mod, label = TRUE,xvar = 'lambda', cex=0.5)

# Plot optimal lambda via cross validation
abline(v=cv.la$lambda.min, add=T)

# Plot optimal lambda via one standard error rule
abline(v=cv.la$lambda.1se, add=T)
```


```
# Show top variable selected by lasso regression to predict response variable
which(coef(lasso.mod)[,100]>0)
```


```
# Test set prediction 
pred_test <- predict(la, s = lambda_best, newx = x[test,])
```


```
# Rooted mean squared error
sqrt(mean((pred_test - ytest)^2))
```


```
# Generate prediction for competition test set for lasso model
x_test = model.matrix(d.test$id~., d.test)[,-1]
pred_lasso = predict(la, s = lambda_best, newx = x_test)
```


```
######################## Boosting With Regression Tress ########################
y.test = d.train[-train,'y']
```


```
# Base boosting model with regression model
boost = gbm(y~.,data=d.train[train,], distribution='gaussian',n.trees = 5000,
            interaction.depth = 6, shrinkage = 0.01, cv.folds = 5)
```


```
# Model summary with order list of relevant predictor variables
summary(boost)
```


```
# Optimal number of tress for model chosen by cross validation
bi = gbm.perf(boost,method="cv")
```


```
# Test set prediction with optimal model
pr.boo = predict(boost,newdata=d.train[-train,],n.trees=bi)
```


```
# Rooted mean squared error
sqrt(mean((pr.boo - y.test)^2))
```


```
# Generate prediction for competition test set for boosting model
pred_boosting = predict(boost,newdata=d.test,n.trees=bi)
```

####################### Generalized Additive Model ############################


```
y.test = d.train[-train,'y']
# Base GAM model with all variables
gam = gam(y~.,data=d.train[train,])
```


```
# Model Summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```

Used relevant variables used by lasso and boosting model above to construct 
GAM model


```
## Variable 1 ##
# Plot variable X4
plot(d.train[train,]$X4, d.train[train,]$y)
```


```
# Run cross validation to choose optimal degrees of freedom for smoothing spline 
fit1 = smooth.spline(d.train[train,]$X4,d.train[train,]$y,cv=TRUE)
fit1
lines(fit1 ,col ="red ",lwd =2)
```


```
# Add variable X4 to GAM model
gam = gam(y~s(X4,8),data=d.train[train,])
```


```
# Model Summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 2 ##
# Plot variable X86
plot(d.train[train,]$X86, d.train[train,]$y)
```


```
# Run cross validation to choose optimal degrees of freedom for smoothing spline
fit1 = smooth.spline(d.train[train,]$X86,d.train[train,]$y,cv=TRUE)
fit1
lines(fit1 ,col ="red ",lwd =2)
```


```
# Add variable X86 to GAM model
gam = gam(y~s(X4,8)+s(X86,11),data=d.train[train,])
```


```
# Model Summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 3 ##
# Plot variable X11
plot(d.train[train,]$X11, d.train[train,]$y)
```


```
# Base local regression model
fit = loess(d.train[train,]$y ~ d.train[train,]$X11, data=d.train,span = 0.1)
fit
lines(fit ,col ="blue ",lwd =2)
```


```
# Choosing optimal span for model
span.seq <- seq(from = 0.1, to = 0.9, by = 0.1)
span = 0.1
testerror = 50000000000
for(i in 1:length(span.seq)) {
  gam = gam(y~s(X4,8)+s(X86,11)+lo(X11, span = span.seq[i]),data=d.train[train,])
  preds <- predict(gam, newdata = d.train[-train,],type="response")
  testerror_i = sqrt(mean((preds - y.test)^2))
  if (testerror_i<testerror){
    testerror = testerror_i
    span = span.seq[i]
  }
}
#span 0.1 selected
span
```


```
# Add variable X11 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1),data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 4 ##
# Plot variable X103
plot(d.train[train,]$X103, d.train[train,]$y)
```


```
# Add varibale X103 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 5 ##
# Plot variable X3
plot(d.train[train,]$X3, d.train[train,]$y)
```


```
# Add variable X3 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```

```


```
## Variable 6 ##
# Plot variable X7
plot(d.train[train,]$X7, d.train[train,]$y)
```


```
# Add variable X7 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 7 ##
# Plot variable X79
plot(d.train[train,]$X79, d.train[train,]$y)
```


```
# Add variable X79 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 8 ##
# Plot variable X43
plot(d.train[train,]$X43, d.train[train,]$y)
```


```
# Add variable X43 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 9 ##
# Plot variable X6
plot(d.train[train,]$X6, d.train[train,]$y)
```


```
# Add variable X6 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```

```


```
## Variable 10 ##
# Plot variable X59
plot(d.train[train,]$X59, d.train[train,]$y)
```


```
# Add variable X59 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 11 ##
# Plot variable X37
plot(d.train[train,]$X37, d.train[train,]$y)
```


```
# Add variable X37 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37,data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 12 ##
# Plot variable X88
plot(d.train[train,]$X88, d.train[train,]$y)
```


```
# Choose optimal span for local regression
span.seq <- seq(from = 0.1, to = 0.9, by = 0.1)
span = 0.1
testerror = 50000000000
for(i in 1:length(span.seq)) {
  gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
            +lo(X88,span = span.seq[i]),data=d.train[train,])
  preds <- predict(gam, newdata = d.train[-train,],type="response")
  testerror_i = sqrt(mean((preds - y.test)^2))
  if (testerror_i<testerror){
    testerror = testerror_i
    span = span.seq[i]
  }
}
#span 0.4 selected
span
```


```
# Add variable X88 to GAM model 
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
          +lo(X88,span=0.4),data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```

############# Add Interaction Variables #################


```
# GAM model with relevant predictors and interaction
gam = gam(y~(X1+X14+X55+X103+X4+X86+X11+X88+X103+X3+X7+X43+X59)^2,data=d.train[train,])
```


```
# Model summary to choose relevant interaction for predictions
summary(gam)
```


```

```


```
## Variable 13 ##
# Plot interaction between variables X14, X1
plot(d.train[train,]$X14*d.train[train,]$X1, d.train[train,]$y)
```


```
# Run cross validation to choose optimal degrees of freedom for smoothing spline
fit1 = smooth.spline(d.train[train,]$X14*d.train[train,]$X1,d.train[train,]$y,cv=TRUE)
fit1
lines(fit1 ,col ="red ",lwd =2)
```


```
# Add interaction between variables X14, X1 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
          +lo(X88,span=0.4)+s(X14*X1,11),data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 14 ##
# Plot interaction between variables X14, X55
plot(d.train[train,]$X14*d.train[train,]$X55, d.train[train,]$y)
```


```
# Run cross validation to choose optimal degrees of freedom for smoothing spline
fit1 = smooth.spline(d.train[train,]$X14*d.train[train,]$X55,d.train[train,]$y,cv=TRUE)
fit1
lines(fit1 ,col ="red ",lwd =2)
```


```
# Add interaction between variables X14, X55 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
          +lo(X88,span=0.4)+s(X14*X1,11)+s(X14*X55,2),data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
#Variable 15
# Plot interaction between variables X1, X55
plot(d.train[train,]$X1*d.train[train,]$X55, d.train[train,]$y)
```


```
# Run cross validation to choose optimal degrees of freedom for smoothing spline
fit1 = smooth.spline(d.train[train,]$X1*d.train[train,]$X55,d.train[train,]$y,cv=TRUE)
fit1
lines(fit1 ,col ="red ",lwd =2)
```


```
# Add interaction between variables X1, X55 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
          +lo(X88,span=0.4)+s(X14*X1,11)+s(X14*X55,2)+s(X1*X55,2),data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
## Variable 16 ##
# Plot interaction between variables X1, X54
plot(d.train[train,]$X1*d.train[train,]$X54, d.train[train,]$y)
```


```
# Choose optimal span for local regression
span.seq <- seq(from = 0.1, to = 0.9, by = 0.1)
span = 0.1
testerror = 50000000000
for(i in 1:length(span.seq)) {
  gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
            +lo(X88,span=0.4)+s(X14*X1,11)+s(X14*X55,2)+s(X1*X55,2)
            +lo(X54,X1,span=span.seq[i]),data=d.train[train,])
  preds <- predict(gam, newdata = d.train[-train,],type="response")
  testerror_i = sqrt(mean((preds - y.test)^2))
  if (testerror_i<testerror){
    testerror = testerror_i
    span = span.seq[i]
  }
}
#span 0.1 selected
span
```


```
# Add interaction between variables X1, X54 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
          +lo(X88,span=0.4)+s(X14*X1,11)+s(X14*X55,2)+s(X1*X55,2)
          +lo(X54,X1,span=0.1),data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
#Variable 17
# Plot interaction between variables X59, X105, X103
plot(d.train[train,]$X59*d.train[train,]$X105*d.train[train,]$X103, d.train[train,]$y)
```


```
# Choose optimal span for local regression
span.seq <- seq(from = 0.1, to = 0.9, by = 0.1)
span = 0.1
testerror = 50000000000
for(i in 1:length(span.seq)) {
  gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
            +lo(X88,span=0.4)+s(X14*X1,11)+s(X14*X55,2)+s(X1*X55,2)
            +lo(X54,X1,span=0.1)+lo(X59,X105,X103,span =span.seq[i]),data=d.train[train,])
  preds <- predict(gam, newdata = d.train[-train,],type="response")
  testerror_i = sqrt(mean((preds - y.test)^2))
  if (testerror_i<testerror){
    testerror = testerror_i
    span = span.seq[i]
  }
}
#span 0.1 selected
span
```


```
# Add interaction between variables X59, X105, X103 to GAM model
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+X103+X3+X7+X79+X43+X6+X59+X37
          +lo(X88,span=0.4)+s(X14*X1,11)+s(X14*X55,2)+s(X1*X55,2)
          +lo(X54,X1,span=0.1)+lo(X59,X105,X103,span =0.1),data=d.train[train,])
```


```
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```

########### Final GAM Model ##############


```
gam = gam(y~s(X4,8)+s(X86,11)+lo(X11,span=0.1)+s(X14*X1,11)+X103
          +s(X14*X55,2)+s(X1*X55,2)+X3+X7+X79+X43+lo(X88,span=0.4)
          +X6+X59+X37+lo(X54,X1,span=0.1)+lo(X59,X105,X103,span =0.1), data=d.train[train,])
# Model summary
summary(gam)
```


```
# Training set prediction
yhat_train = predict(gam, d.train[train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_train - d.train[train,'y'])^2))
```


```
# Test set prediction
yhat_test = predict(gam, d.train[-train,],type="response")
```


```
# Rooted mean squared error
sqrt(mean((yhat_test - y.test)^2))
```


```
######## Final Prediction for Kaggle competition using GAM model ###########
pred_gam = predict(gam,d.test[-1],type="response")
```
