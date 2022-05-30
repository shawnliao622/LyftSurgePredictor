###############Installing_Libraries########

library(tidyverse)
install.packages("readxl")
library(readxl)
library(pacman)
library(e1071)
library(class)
library(tree)
library(randomForest)
library(pROC)
install.packages("car")
library(car)
library(gbm)
library(xgboost)
p_load(caret,fastDummies)

###############Data Pre-Processing#######################

df_train <- read_excel("df_Over_train.xlsx")
df_valid <- read_excel("df_Valid_Lyft.xlsx")

df_train$full_cab_name <- as.factor(df_train$full_cab_name)
df_train$source <- as.factor(df_train$source)
df_train$destination <- as.factor(df_train$destination)
df_train$short_summary <- as.factor(df_train$short_summary)
df_train$time_of_day <- as.factor(df_train$time_of_day) 
df_train$month <- as.factor(df_train$month)  
df_train$NewSurgeMult <- as.factor(df_train$NewSurgeMult) 
df_train$rush_hour <- as.factor(df_train$rush_hour)
df_train$Weekday_Dummy <- as.factor(df_train$Weekday_Dummy)

df_valid$full_cab_name <- as.factor(df_valid$full_cab_name)
df_valid$source <- as.factor(df_valid$source)
df_valid$destination <- as.factor(df_valid$destination)
df_valid$short_summary <- as.factor(df_valid$short_summary)
df_valid$time_of_day <- as.factor(df_valid$time_of_day) 
df_valid$month <- as.factor(df_valid$month) 
df_valid$NewSurgeMult <- as.factor(df_valid$NewSurgeMult) 
df_valid$rush_hour <- as.factor(df_valid$rush_hour)
df_valid$Weekday_Dummy <- as.factor(df_valid$Weekday_Dummy)

###############Principal Component Analysis##############################

train_PCA <- df_train

train_PCA$NewSurgeMult <- as.numeric(train_PCA$NewSurgeMult)  - 1
train_PCA$rush_hour <- as.numeric(train_PCA$rush_hour)  - 1
train_PCA$Weekday_Dummy<- as.numeric(train_PCA$Weekday_Dummy)  -1

str(train_PCA)
num_Variables <- c ("distance", "temperature", "precipProbability", "rush_hour", "Weekday_Dummy", "NewSurgeMult")

pr.out <- prcomp(train_PCA[, num_Variables], scale = TRUE) # create PCA
names(pr.out)

pr.out$center # mean used to scale variables
pr.out$scale  # std used to scale variables
pr.out$rotation # Provides the principal component loadings vectors.

# biplot(pr.out, scale = 0)

pr.var <- pr.out$sdev^2  # compute the variance explained by each principal component
pr.var

pve <- pr.var / sum(pr.var) # compute the proportion of variance explained by each principal component
pve 

# Variance explained by each PC is close # PCA does not work
# Plot the PVE explained by each Principal Component and the cumulative PVE
par(mfrow = c(1, 2))

plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", ylim = c(0, 1),
     type = "b")

plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained", ylim = c(0, 1), type = "b")

###############Linear Probability Model####################################


train_Linear <- df_train
train_Linear$NewSurgeMult <- as.numeric(train_Linear$NewSurgeMult) - 1
test_Linear <- df_valid
test_Linear$NewSurgeMult <- as.numeric(test_Linear$NewSurgeMult) - 1

model_Linear <- lm(NewSurgeMult ~., data = train_Linear)

predict_Linear <- predict(model_Linear , newdata = test_Linear) 

# Warning: prediction from a rank-deficient fit may be misleading
# Check the Multicollinearity

# Compute VIF
vif(model_Linear)

# there are aliased coefficients in the model : Imply that some predictors are perfectly correlated.

summary(model_Linear)

# Coefficient of Destination is NA. It implies the perfect multicollinearity. #Remove destination

model_Linear <- lm(NewSurgeMult ~. - destination , data = train_Linear)
vif(model_Linear)
# precipProbility has GVIF^(1 / 2 * DF) (Equivalent VIF) = 7.6 : GVIF could be applied on categorical variables

predict_Linear_prob <- predict(model_Linear , newdata = test_Linear)   # R-square is only 0.19
predict_Linear <- ifelse(predict_Linear_prob > 0.5, 1, 0 ) #  cutoff = 0.5

table(test_Linear$NewSurgeMult, predict_Linear)

(accuracy_Linear <- mean(test_Linear$NewSurgeMult == predict_Linear)) 
# Accuracy = 0.5512036
(sens_Linear <- sum(test_Linear$NewSurgeMult == 1 & predict_Linear == 1) / sum(test_Linear$NewSurgeMult == 1)) 
# Sensitivity = 0.8313725
(spec_Linear <- sum(test_Linear$NewSurgeMult == 0 & predict_Linear == 0) / sum(test_Linear$NewSurgeMult == 0))
# Specificity = 0.5312892

ROC_Linear <- roc(test_Linear$NewSurgeMult, predict_Linear_prob) #AUC = 0.719

###############Logistic_Model####################################

train_Logi <- df_train
test_Logi <- df_valid

model_Logi <- glm(NewSurgeMult ~ .-destination, data = train_Logi, family = "binomial" )
summary(model_Logi)
predict_Logi_prob <- predict(model_Logi, newdata = test_Logi, type = "response")
predict_Logi <- ifelse(predict_Logi_prob  > 0.5, "1", "0")

(test.confusion <- table(test_Logi$NewSurgeMult, predict_Logi))
(accuracy_Logi <- mean(test_Logi$NewSurgeMult == predict_Logi)) 
(sens_Logi <- sum(test_Logi$NewSurgeMult == 1 & predict_Logi == 1) / sum(test_Logi$NewSurgeMult == 1)) 
(spec_Logi <- sum(test_Logi$NewSurgeMult == 0 & predict_Logi == 0) / sum(test_Logi$NewSurgeMult == 0))

predict_Logi_prop <- predict(model_Logi , newdata = test_Logi) 
ROC_Logi <- roc(test_Linear$NewSurgeMult, predict_Logi_prop) # AUC = 0.720

###############Naive_Bayes############################################

train_NB <- read_excel("df_Over_train.xlsx")
test_NB <- read_excel("df_Valid_Lyft.xlsx")

# For training data  - Remove all the Space
train_NB$short_summary <- gsub('\\s+', '', train_NB$short_summary)

# Drop c0lumns not used for building models
train_NB$full_cab_name <- NULL

train_NB$month <- factor(train_NB$month)
train_NB$short_summary <- factor(train_NB$short_summary)
train_NB$time_of_day <- factor(train_NB$time_of_day)
train_NB$source <- factor(train_NB$source)
train_NB$destination <- factor(train_NB$destination)

# For test data  - Remove all the Space
test_NB$short_summary <- gsub('\\s+', '', test_NB$short_summary)

# Drop columns not used for building models
test_NB$full_cab_name <- NULL

test_NB$month <- factor(test_NB$month)
test_NB$short_summary <- factor(test_NB$short_summary)
test_NB$time_of_day <- factor(test_NB$time_of_day)
test_NB$source <- factor(test_NB$source)
test_NB$destination <- factor(test_NB$destination)

# Model parameters with the best sensitivity

model_NB <- naiveBayes(NewSurgeMult~source+distance+Weekday_Dummy+time_of_day, data=train_NB)
predict_NB <- predict(model_NB, newdata=test_NB)
(CM <- table(predict_NB, test_NB$NewSurgeMult))
(accuracy_NB <- (CM[1,1]+CM[2,2])/sum(CM))
(sens_NB <- CM[2,2]/sum(CM[,2]))
(spec_NB <- CM[1,1]/sum(CM[,1]))

# Plot the lift chart
predicted.probability_NB <- predict(model_NB, newdata = test_NB, type="raw")
PL_NB <- test_NB$NewSurgeMult
prob_NB <- predicted.probability_NB[,2] # Predicted probability of success
df1 <- data.frame(PL_NB, prob_NB)
df1S <- df1[order(-prob_NB),]
df1S$Gains <- cumsum(df1S$PL)
plot(df1S$Gains,type="n",main="Lift Chart - Naive Bayes",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains,col="blue")
abline(0,sum(df1S$PL_NB)/nrow(df1S),lty = 2, col="red")

###############K_Nearest_Neighbors#########

df_train_K <- read_excel("df_Over_train.xlsx")
df_test_K <- read_excel("df_Valid_Lyft.xlsx")

# For training data  - Remove all the Space
df_train_K$short_summary <- gsub('\\s+', '', df_train_K$short_summary)
# For testing data - Remove all the Space
df_test_K$short_summary <- gsub('\\s+', '', df_test_K$short_summary)

# Drop columns not used for building models
df_train_K$full_cab_name <- NULL
df_train_K$source <- NULL
df_train_K$destination <- NULL

df_test_K$full_cab_name <- NULL
df_test_K$source <- NULL
df_test_K$destination <- NULL

df_Dummy <- dummy_cols(df_train_K, select_columns = 
                         c('month','short_summary', 'time_of_day'))
df_Dummy[, c(1,4,6)] <- NULL
df_Scale <- scale(df_Dummy[,-6])
df_Scale <- as.data.frame(df_Scale)
df_Scale$NewSurgeMult <- df_train_K$NewSurgeMult

train_KNN <- df_Scale

# Create dummy variables 
df_Dummy_test <- dummy_cols(df_test_K, select_columns = 
                              c('month','short_summary', 'time_of_day'))
df_Dummy_test[, c(1,4,6)] <- NULL
df_Scale_test <- scale(df_Dummy_test[,-6])
df_Scale_test <- as.data.frame(df_Scale_test)
df_Scale_test$NewSurgeMult <- df_test_K$NewSurgeMult
test_KNN <- df_Scale_test

#Prepping input for KNN 
train_input <- as.matrix(train_KNN[,-21])
train_output <- as.vector(train_KNN[,21])
validate_input <- as.matrix(test_KNN[,-21])

#What is the best K ?
kmax <- 30
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)
Sens_test <- rep(0,kmax)
#
# set.seed(12345) will give K = 15
set.seed(12345)
for (i in 1:kmax){
  prediction <- knn(train_input, train_input,train_output, k=i)
  prediction2 <- knn(train_input, validate_input,train_output, k=i)
  #
  # The confusion matrix for training data is:
  CM1 <- table(df_Scale$NewSurgeMult, prediction)
  # The training error rate is:
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
  # The confusion matrix for test data is: 
  CM2 <- table(df_Scale_test$NewSurgeMult,prediction2)
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
  Sens_test[i] <- (CM2[2,2]/(CM2[2,1]+CM2[2,2]))
}

plot(c(1,kmax),c(0,0.8),type="n", xlab="k",ylab="Ratio")
lines(ER1,col="red")
lines(ER2,col="blue")
lines(Sens_test,col="green")
legend(30, 30, c("Training","Validation", "Sensitivity"),lty=c(1,1), col=c("red","blue", "green"))
z <- which.max(Sens_test)
cat("Maximum Sensitivity k:", z, "\n")
points(z,Sens_test[z],col="red",cex=2,pch=20)

# Validation confusion matrix using best k
prediction_best_k <- knn(train_input, validate_input,train_output, k=z)
(CM_best_k <- table(df_Scale_test$NewSurgeMult,prediction_best_k))
(sens_KNN <- CM_best_k[2,2]/(CM_best_k[2,1]+CM_best_k[2,2]))
(spec_KNN <- CM_best_k[1,1]/(CM_best_k[1,1]+CM_best_k[1,2]))
(accuracy_KNN <- 1- (CM_best_k[1,2]+CM_best_k[2,1])/sum(CM_best_k))

# Creation Lyft chart
prediction <- knn(train_input, validate_input, train_output, k=z, prob=T)
predicted.probability <- attr(prediction, "prob")
prediction_KNN <- knn(train_input, validate_input, train_output, k=z)
predicted.probability_KNN <- ifelse(prediction_KNN ==1, predicted.probability, 1-predicted.probability)

df2 <- data.frame(predicted.probability_KNN,test_KNN$NewSurgeMult)
df2S <- df2[order(-predicted.probability_KNN),]
df2S$Gains <- cumsum(df2S$test_KNN.NewSurgeMult)
plot(df2S$Gains,type="n",main="Lift Chart - KNN",xlab="Number of Cases",ylab="Cumulative Success")
lines(df2S$Gains,col="blue")
abline(0,sum(df2S$test_KNN.NewSurgeMult)/nrow(df2S),lty = 2, col="red")

###############Class Trees#############

df_Over_train <- read_excel("df_Over_train.xlsx")
df_Valid_Lyft <- read_excel("df_Valid_Lyft.xlsx")

df_Over_train$NewSurgeMult<- as.factor(df_Over_train$NewSurgeMult)
df_Over_train$Weekday_Dummy<-as.factor(df_Over_train$Weekday_Dummy)
df_Over_train$rush_hour<-as.factor(df_Over_train$rush_hour) 

df_Valid_Lyft$NewSurgeMult<- as.factor(df_Valid_Lyft$NewSurgeMult)
df_Valid_Lyft$Weekday_Dummy<-as.factor(df_Valid_Lyft$Weekday_Dummy)
df_Valid_Lyft$rush_hour<-as.factor(df_Valid_Lyft$rush_hour)  

set.seed(12345)
str(df_Over_train)

df_Over_train<-na.omit(df_Over_train)
df_Valid_Lyft<-na.omit(df_Valid_Lyft)

tree.lyft=tree(NewSurgeMult~.-source-destination,df_Over_train,
               control=tree.control(nobs=nrow(df_Over_train),minsize=1,mindev=.001))
summary(tree.lyft)
plot(tree.lyft)
text(tree.lyft,pretty=0)

tree.pred=predict(tree.lyft,newdata=df_Valid_Lyft, type="class")
(CMtree = table(df_Valid_Lyft$NewSurgeMult,tree.pred))
(Acctree = (CMtree[1,1]+CMtree[2,2])/sum(CMtree)) #.389

set.seed(5)
cv.lyft=cv.tree(tree.lyft,FUN=prune.misclass)

plot(cv.lyft$size,cv.lyft$dev,type="b")
i=which.min(cv.lyft$dev)
i
z=cv.lyft$size[i]
prune.lyft=prune.misclass(tree.lyft,best=z)
plot(prune.lyft)
text(prune.lyft,pretty=0)

tree.prednew=predict(prune.lyft, df_Valid_Lyft, type="class")
(CMprune = table(df_Valid_Lyft$NewSurgeMult,tree.prednew))
(Accprune = (CMprune[1,1]+CMprune[2,2])/sum(CMprune)) #.389

(sens_CTree <- sum(tree.prednew == "1" & df_Valid_Lyft$NewSurgeMult == "1")/sum(df_Valid_Lyft$NewSurgeMult == "1"))
(spec_CTree <- sum(tree.prednew == "0" & df_Valid_Lyft$NewSurgeMult == "0")/sum(df_Valid_Lyft$NewSurgeMult == 0))

# Now for the ROC curve and Lift Chart
predicted.prob_tree= predict(prune.lyft, df_Valid_Lyft, type="vector")[,2]
df_Valid_Lyft$NewSurgeMult
roc_rose <- plot(roc(df_Valid_Lyft$NewSurgeMult, predicted.prob_tree), print.auc = TRUE, 
                 col = "red", print.auc.y = .3, legacy.axes=T) #AUC is 0.56

###############Ensemble Techniques############


train_boosting<-read_excel("df_Over_train.xlsx")
test_boosting<-read_excel("df_Valid_Lyft.xlsx")

names <- c("full_cab_name","source","destination", "short_summary","time_of_day")
train_boosting[,names] <- lapply(train_boosting[,names] , factor)
str(train_boosting)

set.seed(12345)

#Grid Search : Link for reference = http://uc-r.github.io/gbm_regression#gbm

hyper_grid <- expand.grid(
  shrinkage = c(.01, .05, .1),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 7, 10),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

for(i in 1:nrow(hyper_grid)) {
  # reproducibility
  set.seed(123)
  # train model
  gbm.tune <- gbm(formula = NewSurgeMult~ . ,distribution = "bernoulli",data =train_boosting ,n.trees = 500,
                  interaction.depth = hyper_grid$interaction.depth[i],
                  shrinkage = hyper_grid$shrinkage[i],
                  n.minobsinnode = hyper_grid$n.minobsinnode[i],
                  bag.fraction = hyper_grid$bag.fraction[i],
                  train.fraction = .75,
                  n.cores = NULL, # will use all cores by default
                  verbose = FALSE
  )
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

model_boosting=gbm(NewSurgeMult~.,data=train_boosting,distribution="bernoulli",n.trees=498,interaction.depth=7, shrinkage = 0.1, n.minobsinnode = 7, bag.fraction = 0.65)
summary(model_boosting)

predict.boost=predict(model_boosting,newdata=test_boosting,n.trees=498,type="response")
predict.boost
predicted.boost <- ifelse(predict.boost>=0.5,1,0)
(c = table(predicted.boost,test_boosting$NewSurgeMult))
(acc_boosting = (c[1,1]+c[2,2])/sum(c))
(sens_boosting <- sum(predicted.boost == "1" & test_boosting$NewSurgeMult == "1")/sum(test_boosting$NewSurgeMult == "1"))
(spec_boosting <- sum(predicted.boost== "0" & test_boosting$NewSurgeMult == "0")/sum(test_boosting$NewSurgeMult == 0))

###############XGBooost Model ------------------------------

train<-read_excel("df_Over_train.xlsx")
train<-select(train,-full_cab_name)
test<-read_excel("df_Valid_Lyft.xlsx")
test<-select(test,-full_cab_name)

train_XG <- dummy_cols(train, select_columns = c("source","destination", "short_summary","time_of_day"), remove_selected_columns = TRUE)
test_XG <- dummy_cols(test, select_columns = c("source","destination", "short_summary","time_of_day"), remove_selected_columns = TRUE)
set.seed(12345)
label=train_XG$NewSurgeMult
data_lyft = as.matrix(train_XG[,-7])

#Hypertune Reference Xgboost : https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/

bstCV <- xgb.cv(data = data_lyft, label = label, nfold = 5,
              nrounds = 1000, objective = "binary:logistic",
              early_stopping_rounds = 2, maximize = FALSE)

model_xgbst <- xgboost(data = data_lyft, label = label, max.depth = 1, eta = 0.7, nround = 921, objective = "binary:logistic")
summary(model_xgbst)

labelT=test_XG$NewSurgeMult
datalyftT = as.matrix(test_XG[,-7])
pred_Xgboost <- predict(model_xgbst, datalyftT)
predicted_Xgboost <- ifelse(pred_Xgboost>0.5,1,0)

(c_XGboost = table(labelT,predicted_Xgboost))
(accuracy_Xgboost = (c_XGboost[1,1]+c_XGboost[2,2])/sum(c_XGboost))
(sens_Xgboost <- sum(predicted_Xgboost == "1" & labelT == "1")/sum(labelT == "1"))
(spec_Xgboost <- sum(predicted_Xgboost == "0" & labelT == "0")/sum(labelT == 0))
roc_Xgboost <- plot(roc(yhat.test, predicted_Xgboost), print.auc = TRUE, col = "red")

par(bg = "s")

dfxgboost <- data.frame(predicted_Xgboost,test_XG$NewSurgeMult)
df2S <- dfxgboost[order(-predicted_Xgboost),]
df2S$Gains <- cumsum(df2S$test_XG.NewSurgeMult)
plot(df2S$Gains,type="n",main="Test Data Lift Chart - XGBoost",xlab="Number of Cases",ylab="Cumulative Success")
lines(df2S$Gains, col="deeppink4")
abline(0,sum(df2S$test_XG.NewSurgeMult)/nrow(df2S),lty = 2, col="black")

importance <- xgb.importance(model = model_xgbst)
head(importance)
xgb.plot.importance(importance_matrix = importance)+ title("Variable Importace (Source & Destination)")

train<-select(train,-source,-destination)
test<-select(test,-source,-destination)

train_XG1 <- dummy_cols(train, select_columns = c("short_summary","time_of_day"), remove_selected_columns = TRUE)
test_XG1 <- dummy_cols(test, select_columns = c("short_summary","time_of_day"), remove_selected_columns = TRUE)

set.seed(12345)
label=train_XG1$NewSurgeMult
data_lyft1 = as.matrix(train_XG1[,-7])

model_xgbst1 <- xgboost(data = data_lyft1, label = label, max.depth = 1, eta = 0.7, nround = 500, objective = "binary:logistic")
summary(model_xgbst1)
labelT=test_XG1$NewSurgeMult
datalyftT = as.matrix(test_XG1[,-7])
pred_Xgboost1 <- predict(model_xgbst1, datalyftT)
predicted_Xgboost1 <- ifelse(pred_Xgboost1>0.5,1,0)
(c_XGboost1 = table(labelT,predicted_Xgboost1))
(accuracy_Xgboost1 = (c_XGboost1[1,1]+c_XGboost1[2,2])/sum(c_XGboost1))
(sens_Xgboost1 <- sum(predicted_Xgboost1 == "1" & labelT == "1")/sum(labelT == "1"))
(spec_Xgboost1 <- sum(predicted_Xgboost1 == "0" & labelT == "0")/sum(labelT == 0))

importance <- xgb.importance(model = model_xgbst1)
head(importance)
xgb.plot.importance(importance_matrix = importance)+ title ("Variable Importance")

###############CombinedModelsMethod#####
pred_XGBOOST <- predicted_Xgboost
pred_LINEAR <- predict_Linear
pred_LOGI <- predict_Logi
Actuals <- df_Valid_Lyft$NewSurgeMult
  
dfcombined <- data.frame(pred_LINEAR, pred_XGBOOST, predict_Logi)
View(dfcombined)

col <- apply(dfcombined,1,function(x) names(which.max(table(x))))
#Source : https://stackoverflow.com/questions/19982938/find-the-most-frequent-value-by-row
View(col)

newdf<-data.frame(Actuals, col,pred_LINEAR, pred_XGBOOST, predict_LOGI)
View(newdf)


(c_newdf = table(Actuals,col))
(accuracy_newdf= (c_newdf[1,1]+c_newdf[2,2])/sum(c_newdf))

(sens_comb <- sum(col == "1" & Actuals == "1")/sum(Actuals == "1"))
## Specificity
(spec_comb <- sum(col == "0" & Actuals == "0")/sum(Actuals == 0))

#####################ROC_Curves - All ############## 

par(bg = "gray88")
roc_linear <- plot(ROC_Linear, print.auc = TRUE, col = "deeppink4", print.auc.y = .8, legacy.axes=T, asp=NA)
roc_log <- plot(ROC_Logi, print.auc = TRUE, 
                 col = "deeppink", print.auc.y = .7, legacy.axes=T, asp=NA, add=TRUE)+ title ("ROC Curves(Regression Models)", line = 2.5, adj = 0.5)
legend(0.6, 0.4, c("Linear Model", "Logistic Model"),lty=c(1,1), col=c("deeppink4","deeppink"), cex=0.5, text.font=1.5, title="Legend")


ROC_NB <- plot(roc(PL_NB, prob_NB), print.auc = TRUE, col = "violetred", print.auc.y = .9, print.auc.x = .95 ,legacy.axes=T)
ROC_KNN <- plot(roc(test_KNN$NewSurgeMult, predicted.probability_KNN), print.auc = TRUE, print.auc.y = .83, print.auc.x = .95,
                legacy.axes=T, col = "purple", add=TRUE)
roc_CTree <- plot(roc(df_Valid_Lyft$NewSurgeMult, predicted.prob_tree), print.auc = TRUE, 
                 col = "indianred4", print.auc.y = .75, print.auc.x = .95,legacy.axes=T, add=T) + title ("ROC Curves(Trees & NB/KNN)", line = 2.5, adj = 0.5)
legend(0.4, 0.4, c("NB", "KNN", "CTree"),lty=c(1,1), col=c("violetred","purple", "indianred4"), cex=0.5, text.font=1.5, title="Legend")

roc_Boosting <- plot(roc(test_XG$NewSurgeMult, predict.boost), print.auc = TRUE, col = "deeppink4", print.auc.y = .50, legacy.axes=T)
roc_Xgboost <- plot(roc(test_XG$NewSurgeMult, pred_Xgboost), print.auc = TRUE, col = "deeppink", add=T, print.auc.y = .40) + title ("ROC Curves (Ensemble Techniques)", line = 2.5, adj = 0.5)
legend(0.9, 0.9, c("Boosting", "Xgboost"),lty=c(1,1), col=c("violetred","deeppink2", "gray37"), cex=0.5, text.font=1.5, title="Legend")

#######################################THE END######################################

