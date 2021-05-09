# Wine Project
# Machine Learning class

##################################################
#import data
library(readxl)
winequality_white <- read_excel("~/Desktop/ml/winequality-white.xlsx")
white=winequality_white
#add id index
white$id<-seq.int(nrow(white))
save(white, file="white.Rdata")
#split data
spec = c(train=0.7, test=0.2, validate=0.1)
g=sample(cut(
  seq(nrow(white)),
  nrow(white)*cumsum(c(0, spec)),
  labels = names(spec)
))
whi=split(white, g)
#save new datasets
d.test<-whi$test
d.val<-whi$validate
d.train<-whi$train
save(d.test, file="d.test.Rdata")
save(d.val, file="d.val.Rdata")
save(d.train, file="d.train.Rdata")
##################################################

load("~/Desktop/ml/white.Rdata")
load("~/Desktop/ml/d.test.Rdata")
load("~/Desktop/ml/d.val.Rdata")
load("~/Desktop/ml/d.train.Rdata")
train_set=rbind(d.train,d.val)


#features for overall dataset
table(white$quality)
hist(white$quality)
summary(white$quality)
wine.var<-white[,1:12]
round(cor(wine.var),2)




#model selection
library(ISLR)
library(leaps)
regfit.best=regsubsets(quality ~ .-id,data=d.train, nvmax =11)
reg.sum=summary(regfit.best)
#using adjusted R2
which.max(reg.sum$adjr2)
#using the Validation set approach
val.mat=model.matrix(quality ~ .-id,data=d.val)
val.errors=rep(NA,11)
for(i in 1:11){
  coefi=coef(regfit.best,id=i)
  pred=val.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((d.val$quality-round(pred))^2) 
}
which.min(val.errors)

coef(regfit.best,which.max(reg.sum$adjr2))
coef(regfit.best,which.min(val.errors))
#The model with 9 predictors has the maximum adjusted R squared.
#The model with 3 predictors has the smallest validation set error.

#full
lm.fit1=lm(quality ~ .-id,data=d.train)
lm.pred1 <- predict(lm.fit1, newdata = d.val)
mean(round(lm.pred1)==d.val$quality)
#3 predictors
lm.fit2=lm(quality ~ volatile.acidity+residual.sugar+alcohol,data=d.train)
lm.pred2 <- predict(lm.fit2, newdata = d.val)
mean(round(lm.pred2)==d.val$quality)
#9 predictors
lm.fit4=lm(quality ~ fixed.acidity + volatile.acidity + residual.sugar + free.sulfur.dioxide + density + total.sulfur.dioxide + pH + sulphates + alcohol,data=d.train)
lm.pred4 <- predict(lm.fit4, newdata = d.val)
mean(round(lm.pred4)==d.val$quality)
#Both the full model and the model with 3 predictors have the highest accuracy for prediction. Therefor, with the purpose of increasing accuracy and preventing overfitting, we consider model 2 as the best model.
#final linear regression model
lm.fit=lm(quality ~ volatile.acidity+residual.sugar+alcohol,data=train_set)
summary(lm.fit)
#prediction performance for test data
lm.pred=predict(lm.fit, newdata = d.test)
#error rate
err_lm=mean(round(lm.pred4)!=d.test$quality)
err_lm
#Mean squared error
mse_lm=mean((round(lm.pred4)-d.test$quality)^2)
mse_lm
#error rate is 0.6255102 and mse is 1.182653.

################
#2. Ridge Regression
################

library(glmnet)
library(Matrix)
train.y=d.train$quality
val.y=d.val$quality
test.y=d.test$quality

train.x<-model.matrix(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides +free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol,d.train)[,-1]
val.x<-model.matrix(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides +free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol,d.val)[,-1]
test.x<-model.matrix(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides +free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol,d.test)[,-1]

grid<-10^seq(10, -2, length = 100)
train.ridge<- glmnet(train.x, train.y, alpha=0, lambda=grid)
#lambda by cross validation
cv.ridge<-cv.glmnet(val.x, val.y, alpha=0)
bestlam.ridge <- cv.ridge$lambda.min
bestlam.ridge
pred.ridge<-predict(train.ridge, s=bestlam.ridge, newx=test.x)
#error rate
mean(round(pred.ridge)!=test.y)
#MSE
mean((round(pred.ridge)-test.y)^2)

################
# 3.Lasso Regression
################

library(glmnet)
library(Matrix)

grid=10^seq(10,-2,length=100)
lasso.fit=glmnet(data.matrix(d.train[,1:11]),d.train$quality,alpha=1,lambda=grid)
#Use cross validation to choose lamda
set.seed(1)
cv.out=cv.glmnet(data.matrix(d.val[,1:11]),d.val$quality,alpha=1)
bestlam=cv.out$lambda.min
bestlam
# lamda of 0.005527695 results in the smallest CV error.

#refit the lasso regression on the whole train set(train+val)
lasso.fit=glmnet(data.matrix(train_set[,1:11]),train_set$quality,alpha=1,lambda=grid)
lasso.coef=predict(lasso.fit,type="coefficients",s=bestlam)[1:11,]
lasso.coef[lasso.coef!=0]
#Here we see that 3 of the 11 coefficient estimates are exactly zero. So the lasso model with lamda chosen by cross-validation contains only 8 variables.

lasso.pred=predict(lasso.fit,s=bestlam ,newx=data.matrix(d.test[,1:11]))
err_lasso=mean(round(lasso.pred)==d.test$quality)
err_lasso
#error rate is 0.4755102
mse_lasso=mean((round(lasso.pred)-d.test$quality)^2)
mse_lasso
#MSE is 0.6469388

################
# 4.KNN
################

library(class)
knn.train.x<-cbind(d.train$fixed.acidity, d.train$volatile.acidity, d.train$citric.acid, d.train$residual.sugar, d.train$chlorides, d.train$free.sulfur.dioxide, d.train$total.sulfur.dioxide,d.train$density, d.train$pH, d.train$sulphates, d.train$alcohol)
knn.val.x<-cbind(d.val$fixed.acidity, d.val$volatile.acidity, d.val$citric.acid, d.val$residual.sugar, d.val$chlorides, d.val$free.sulfur.dioxide, d.val$total.sulfur.dioxide,d.val$density, d.val$pH, d.val$sulphates, d.val$alcohol)
knn.test.x<-cbind(d.test$fixed.acidity, d.test$volatile.acidity, d.test$citric.acid, d.test$residual.sugar, d.test$chlorides, d.test$free.sulfur.dioxide, d.test$total.sulfur.dioxide, d.test$density, d.test$pH, d.test$sulphates, d.test$alcohol)

knn.train.y=d.train$quality
knn.val.y=d.val$quality
knn.test.y=d.test$quality

set.seed(1)
knn.val.pred=knn(knn.train.x, knn.val.x, knn.train.y, k=1)
mean(knn.val.pred != knn.val.y)
#test error=0.4897959

#standardized knn
std.train.x=scale(d.train[,-13][,-12])
std.val.x=scale(d.val[,-13][,-12])
std.test.x=scale(d.test[,-13][,-12])
s.train.y=d.train$quality
s.val.y=d.val$quality
s.test.y=d.test$quality
s.knn.pred=knn(std.train.x,std.val.x,s.train.y,k=1)
mean(s.val.y!=s.knn.pred)
#test error=0.3734694

#refit final model
s.knn.yhat=knn(std.train.x,std.test.x,s.train.y,k=1)
mean(s.test.y!=s.knn.yhat)
#test error=0.3908163
#mean((d.test$quality-is.numeric(s.knn.yhat))^2)
#MSE is 34.95408

################
# 5.Decision Tree
################

library(tree)
tree<-tree(quality~.-id, data=d.train)
summary(tree)
plot(tree)
text(tree ,pretty =0)

tree.pred=predict(tree,d.val)
mean(round(tree.pred)!=d.val$quality)
#pruning:C-V
set.seed(1)
cvtree<-cv.tree(tree,FUN=prune.tree)
plot(cvtree$size, cvtree$dev, type='b')
size=cvtree$size[which.min(cvtree$dev)]
size
#pruned tree with best size 
prune.fit=prune.tree(tree,best=size)
prune.pred<-predict(prune.fit, d.val)
#error rate
mean(round(prune.pred)!=d.val$quality)

#final model
tree<-tree(quality~.-id, data=train_set)
yhat=predict(tree,d.test)
mean(round(prune.pred)!=d.val$quality)
mean((round(prune.pred)-d.val$quality)^2)

################
# 6.Random Forest
################
library(randomForest)
set.seed (1)
#tune mtry
rf.train.x<-cbind(train_set$fixed.acidity, train_set$volatile.acidity, train_set$citric.acid, train_set$residual.sugar, train_set$chlorides, train_set$free.sulfur.dioxide, train_set$total.sulfur.dioxide,train_set$density, train_set$pH, train_set$sulphates, train_set$alcohol)
mtry.best=tuneRF(rf.train.x,train_set$quality,ntreeTry = 500,stepFactor=2, improve=0.05,trace=TRUE,plot=TRUE)
#mtry.best=3 with smallest OOB error 0.3669797
#tune ntree
library(doParallel)
library(caret)
cores <- makeCluster(detectCores()-1)
registerDoParallel(cores = cores)
control=trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        search = 'grid')
tunegrid <- expand.grid(.mtry = 3)
modellist <- list()
for (ntree in c(100, 500, 1000, 1500,2000,2500,3000,3500,4000,4500,5000)) {
  set.seed(123)
  fit <- train(quality ~ . -id,data=train_set, method="rf", metric="RMSE", tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
#ntree 200 mean RMSE 0.6112882
rf.fit=randomForest(quality ~ .-id,data=train_set,mtry=3,ntree=2000,importance =TRUE)
rf.fit
#prediction
rf.pred = predict(rf.fit ,newdata=d.test)
err_rf=mean(round(rf.pred)!=d.test$quality)
err_rf
# error rate is 0.3132653
mse_rf=mean((round(rf.pred)-d.test$quality)^2)
mse_rf
#MSE is  0.427551

#check importance
importance(rf.fit)        
varImpPlot(rf.fit) 
