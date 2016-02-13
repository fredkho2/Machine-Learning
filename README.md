# Machine-Learning
Machine Learning RMD
Fred
February 13, 2016
################################################################################
## File: Machine Learning Final project by Frederic K                  
## Exec Summary: We are interested in predicting the classe on a gym
## dataset, there are 5 different classes: A, B, C, D, E  
################################################################################

## Libraries
set.seed(1234)
suppressMessages(library(caret));suppressMessages(library(rpart));suppressMessages(library(rattle))
suppressMessages(library(randomForest));suppressMessages(library(rpart.plot));suppressMessages(library(RColorBrewer)) 

## Downloading Files, we were provided with a testing file that will be used for the prediction for the quizz, we named it final_testing
training <- read.csv("pml-training.csv",na.strings = c("NA","DIV/0!",""))
final_testing <- read.csv("pml-testing.csv",na.strings = c("NA","DIV/0!",""))

## Partitioning the training data set into a training_part and a test_part set
inTrain <- createDataPartition(y=training$classe,p=0.7,list = FALSE)
training_part <- training[inTrain,]
testing_part <- training[-inTrain,]
dim(training_part);dim(testing_part);
## [1] 13737   160
## [1] 5885  160
## Data Cleaning
## Step 1: Remove the data with low variances as they do not contribute to the explanation of the problem
Data_With_Low_Var <- nearZeroVar(training_part) 
training_part_without_low_var<-training_part[,-Data_With_Low_Var]

## Step 2: Remove the data with too many NAs, we did arbitrarily choose a 0.6 threshold
Col_training_part_NA <- which(as.numeric(colSums(is.na(training_part_without_low_var))/nrow(training_part_without_low_var))>0.6)
training_part_without_low_var_and_NAs <- training_part_without_low_var[,-Col_training_part_NA]

## Step 3: We will remove the identity and timestamps as it is not needed and creates errors in the predictions
training_part_without_low_var_and_NAs_and_IDs <- training_part_without_low_var_and_NAs[,-c(1:6)]

## Let's apply the same transformations to testing_part and final_testing as we want our data to be consistent
Remaining_col_names <- colnames(training_part_without_low_var_and_NAs_and_IDs)

testing_part <- testing_part[Remaining_col_names]
Remaining_col_names_2<- Remaining_col_names[-53]
final_testing<- final_testing[Remaining_col_names_2]

## Cross Validation

T_Control_def <- trainControl(method = "cv",number = 5,verboseIter = FALSE,preProcOptions = "pca",allowParallel = TRUE)

## Other tests were performed but in order for the code to be easily readable we only showed SVM and RF
## Trying random forest which has an accuracy of 99% in that case
## Disclaimer: fit2 with  train is very slow compared to fit2 with function randomforest

fit2 <- train(classe~.,data = training_part_without_low_var_and_NAs_and_IDs,method = "rf", trControl = T_Control_def)
## fit2 <- randomForest(classe~.,data =  training_part_without_low_var_and_NAs_and_IDs)
pred2 <- predict(fit2, testing_part)
confusionMatrix(pred2,testing_part$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    8    0    0    0
##          B    2 1128    1    1    1
##          C    0    3 1020    5    3
##          D    0    0    5  956    3
##          E    0    0    0    2 1075
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9903   0.9942   0.9917   0.9935
## Specificity            0.9981   0.9989   0.9977   0.9984   0.9996
## Pos Pred Value         0.9952   0.9956   0.9893   0.9917   0.9981
## Neg Pred Value         0.9995   0.9977   0.9988   0.9984   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1917   0.1733   0.1624   0.1827
## Detection Prevalence   0.2855   0.1925   0.1752   0.1638   0.1830
## Balanced Accuracy      0.9985   0.9946   0.9959   0.9950   0.9966
## Accuracy of the SVM Radial is 93%
fit3 <- train(classe ~ ., data = training_part_without_low_var_and_NAs_and_IDs, method = "svmRadial", trControl= T_Control_def)
## Loading required package: kernlab
## 
## Attaching package: 'kernlab'
## The following object is masked from 'package:ggplot2':
## 
##     alpha
pred3 <- predict(fit3, testing_part)
confusionMatrix(pred3,testing_part$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1665   97    7    6    2
##          B    5 1013   34    4   15
##          C    4   27  965  105   46
##          D    0    0   20  846   31
##          E    0    2    0    3  988
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9307         
##                  95% CI : (0.9239, 0.937)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9121         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.8894   0.9405   0.8776   0.9131
## Specificity            0.9734   0.9878   0.9625   0.9896   0.9990
## Pos Pred Value         0.9370   0.9458   0.8413   0.9431   0.9950
## Neg Pred Value         0.9978   0.9738   0.9871   0.9763   0.9808
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2829   0.1721   0.1640   0.1438   0.1679
## Detection Prevalence   0.3020   0.1820   0.1949   0.1524   0.1687
## Balanced Accuracy      0.9840   0.9386   0.9515   0.9336   0.9560
## Providing the answers to the final_testing data set

Pred_final_testing <- predict(fit2,final_testing)
Pred_final_testing
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
