#author :kumar abhinav
setwd("C:/Users/Lenovo/Downloads/tds/Erobern")
library(readr)
train <- read_csv("C:/Users/Lenovo/Downloads/tds/Erobern/train.csv")
test <- read_csv("C:/Users/Lenovo/Downloads/tds/Erobern/test.csv")
cor(train)
regmodel <- lm(Energy ~ ., data = train)
summary(regmodel)
#Adjusted R-squared:  0.1425 :very low
par(mfrow=c(2,2))
plot (regmodel)
#heteroskedasticity removal 
regmodel <- update(regmodel, log(Energy)~.)
summary(regmodel)
#Adjusted R-squared:  0.2395 
regpred <- predict(regmodel, test)
regpred <- exp(regpred)

library(Metrics)
rmse(actual = train$Energy,predicted = regpred)
#[1] 107.4983
main_predict <- predict(regpred, newdata = test)
sub_file <- data.frame(Observation = test$Observation, Energy = main_predict)
write.csv(sub_file, 'regression model .csv')
#===============================================================================================

#building decision trees model
library(rpart)
library(e1071)
library(rpart.plot)
library(caret)
#setting the tree control parameters
fitControl <- trainControl(method = "cv", number = 5)
cartGrid <- expand.grid(.cp=(1:50)*0.01)
#decision tree
tree_model <- train(Energy ~ ., data = train, method = "rpart", trControl = fitControl, tuneGrid = cartGrid)
print(tree_model)
#The final value used for the model was cp = 0.01.
main_tree <- rpart(Energy ~ ., data =train, control = rpart.control(cp=0.01))
prp(main_tree)
pre_score <- predict(main_tree, type = "vector")
rmse(actual = train$Energy,predicted = pre_score)
#[1] 99.77347
#which is actually a improvement of above model
main_predict <- predict(main_tree, newdata = test)
sub_file <- data.frame(Observation = test$Observation, Energy = main_predict)
write.csv(sub_file, 'decision tree algo model.csv')

#===============================================================================================
#understanding data better
library(ggplot2)
ggplot(train, aes(x= T1, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T1")
+ ylab("Energy") + ggtitle("T1 vs Energy")

ggplot(train, aes(x= T2, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T2")
+ ylab("Energy") + ggtitle("T2 vs Energy")

ggplot(train, aes(x= T3, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T3")
+ ylab("Energy") + ggtitle("T3 vs Energy")

ggplot(train, aes(x= T4, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T4")
+ ylab("Energy") + ggtitle("T4 vs Energy")

ggplot(train, aes(x= T5, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T5")
+ ylab("Energy") + ggtitle("T5 vs Energy")

ggplot(train, aes(x= T6, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T6")
+ ylab("Energy") + ggtitle("T6 vs Energy")

ggplot(train, aes(x= T7, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T7")
+ ylab("Energy") + ggtitle("T7 vs Energy")

ggplot(train, aes(x= T8, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T8")
+ ylab("Energy") + ggtitle("T8 vs Energy")

ggplot(train, aes(x= T9, y = Energy)) + geom_point(size = 2.5, color="navy") 
+ xlab("T9")
+ ylab("Energy") + ggtitle("T9 vs Energy")

ggplot(train, aes(x=RH_1 , y = Energy)) + geom_point(size = 2.5, color="darkgreen")
+ xlab("RH_1")
+ ylab("Energy") + ggtitle("RH_1 vs Energy")

ggplot(train, aes(x= RH_2, y = Energy)) + geom_point(size = 2.5, color="darkgreen")
+ xlab("RH_2")
+ ylab("Energy") + ggtitle("RH_2 vs Energy")

ggplot(train, aes(x= RH_3, y = Energy)) + geom_point(size = 2.5, color="darkgreen") 
+ xlab("RH_3")
+ ylab("Energy") + ggtitle("RH_3 vs Energy")

ggplot(train, aes(x= RH_4, y = Energy)) + geom_point(size = 2.5, color="darkgreen") 
+ xlab("RH_4")
+ ylab("Energy") + ggtitle("RH_4 vs Energy")

ggplot(train, aes(x=RH_5, y = Energy)) + geom_point(size = 2.5, color="darkgreen") 
+ xlab("RH_5")
+ ylab("Energy") + ggtitle("RH_5 vs Energy")

ggplot(train, aes(x= RH_6, y = Energy)) + geom_point(size = 2.5, color="darkgreen") 
+ xlab("RH_6")
+ ylab("Energy") + ggtitle("RH_6 vs Energy")

ggplot(train, aes(x= RH_7, y = Energy)) + geom_point(size = 2.5, color="darkgreen") 
+ xlab("RH_7")
+ ylab("Energy") + ggtitle("RH_7 vs Energy")

ggplot(train, aes(x= RH_8,y = Energy)) + geom_point(size = 2.5, color="darkgreen") 
+ xlab("RH_8")
+ ylab("Energy") + ggtitle("RH_8 vs Energy")

ggplot(train, aes(x= RH_9, y = Energy)) + geom_point(size = 2.5, color="darkgreen") 
+ xlab("RH_9")
+ ylab("Energy") + ggtitle("RH_9 vs Energy")

#=======================================================================================
library(corrplot)
cr<-cor(train)
corrplot(cr,method = "circle")
corrplot(cr,method="pie")
corrplot(cr,method="color")



library(xgboost)
library(dplyr)
library(caret)
test$Energy<-1
labels <- train$Energy 
ts_label <- test$Energy
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1
dtrain<-dtrain[,-c(26)]
dtest<-dtest[,-c(26)]
new_tr <- model.matrix(~.+0,data = dtrain) 
new_ts <- model.matrix(~.+0,data = dtest)

d_train <- xgb.DMatrix(data = new_tr,label = labels) 
d_test <- xgb.DMatrix(data = new_ts,label=ts_label)
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.3,
               gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = d_train, nrounds = 1500,
                 nfold = 5, showsd = T, stratified = T, print.every.n = 10,
                 early.stop.round = 20, maximize = F)
#Best iteration:
#  [912]	train-rmse:2.840677+0.107496	test-rmse:71.654700+2.948907

xgb1 <- xgb.train (params = params, data = d_train, nrounds = 912, 
                   watchlist = list(val=d_test,train=d_train), print.every.n = 10,
                   early.stop.round = 10, maximize = F , eval_metric = "rmse")

xgbpred <- predict (xgb1,d_test)
head(xgbpred)

sub_file <- data.frame(Observation = test$Observation,Energy = xgbpred)
write.csv(sub_file, 'xgb model with default param 2.csv')
rmse(actual = train$Energy,predicted =xgbpred)
#[1] 130.041

#important gains
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:25]) 
#========================================================================================
#models other than linear regression
CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=5,
                                 verboseIter=FALSE)

lambdas <- seq(1,0,-0.001)
# train model
library(glmnet)
set.seed(123)
model_ridge <- train(x=dtrain,y=train$Energy,
                     method="glmnet",
                     metric="RMSE",
                     maximize=FALSE,
                     trControl=CARET.TRAIN.CTRL,
                     tuneGrid=expand.grid(alpha=0, # Ridge regression
                                          lambda=lambdas))

ggplot(data=filter(model_ridge$result,RMSE<150)) +
  geom_line(aes(x=lambda,y=RMSE))
mean(model_ridge$resample$RMSE)
#[1] 93.8317
ridgepred <- predict(model_ridge,dtest)
rmse(actual = train$Energy,predicted =ridgepred)
#[1] 106.705
sub_file <- data.frame(Observation = test$Observation,Energy = ridgepred)
write.csv(sub_file, 'ridge reg model.csv')
#======================================================================================
set.seed(123)  # for reproducibility
model_lasso <- train(x=dtrain,y=train$Energy,
                     method="glmnet",
                     metric="RMSE",
                     maximize=FALSE,
                     trControl=CARET.TRAIN.CTRL,
                     tuneGrid=expand.grid(alpha=1,  # Lasso regression
                                          lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001),
                                                   0.00075,0.0005,0.0001)))
model_lasso
#lambda   RMSE      Rsquared   MAE     
#0.00010  93.54672  0.1409547  53.13319
#0.00050  93.54672  0.1409547  53.13319
#0.00075  93.54672  0.1409547  53.13319
#0.00100  93.54672  0.1409547  53.13319
#0.00200  93.54672  0.1409547  53.13319
#0.00300  93.54672  0.1409547  53.13319
#0.00400  93.54674  0.1409544  53.13316
#0.00500  93.54681  0.1409534  53.13289
#0.00600  93.54687  0.1409522  53.13254
#0.00700  93.54691  0.1409514  53.13223
#0.00800  93.54696  0.1409502  53.13192
#0.00900  93.54701  0.1409492  53.13160
#0.01000  93.54707  0.1409478  53.13121
#0.05000  93.55039  0.1408670  53.09206
#0.10000  93.56628  0.1406015  53.04209
#1.00000  94.54216  0.1254623  53.48689
mean(model_lasso$resample$RMSE)
#[1] 93.54672
lassopred <- (predict(model_lasso,newdata=dtest))
rmse(actual = train$Energy,predicted =lassopred)
#[1] 108.0663
sub_file <- data.frame(Observation = test$Observation,Energy = lassopred)
write.csv(sub_file, 'lasso reg model.csv')





