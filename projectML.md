# Automatic recognition of posture in weightlifting
Felipe Lopez  

## Introduction

Poor posture is a very common condition. Posture issues are commonly caaused by muscular imbalances, meaning that some muscles are too tight or too weak.
Poor posture can result in several conditions, the most common is back pain, which according to the [American Chiropractic Association](http://www.dlchiropractors.com/uploads/5/8/0/6/58063233/back_pain_facts_and_statistics.pdf) is experienced by 70 to 85 percent of people at
some point. Posture problems can be greatly improved by performing the right weightlifting exercises. If, however, an inappropriate posture is maintained
while lifting weights, posture problems get even worse. Assistance on weightlifting is often provided by personal trainers. However, measurements
gathered with wearable devices may be used as  an alternative for automated assistance and posture correction. This report describes a machine learning
approach for automatic quality recognition of posture in weightlifting. Data was gathered from accelerometers mounted on the belt, forearm, arm, and
dumbbell of six participants to predict the manner in which they exercise.

## Data

Data were obtained from the Weight Lifting Exercises Data, part of the [Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har) project.
Numerous variables are monitored during unilateral dumbell bicep curls along with the variable *classe* (A-E), which indicates whether the exercise was
performed (A) exactly according to specification, (B) throwing elbows to the front, (C) lifting dumbbell only halfway, (C) lowering dumbbell only halfway,
or (E) throwing hips to the front.

As a first step, the training and testing sets were loaded and factor variables were identified

```r
options(scipen = 1); options(digits=3)
library(data.table); library(dplyr); library(ggplot2); library(lubridate)
library(caret); library(AppliedPredictiveModeling); library(randomForest)
library(earth); library(rpart)
# Acquire training and testing data
trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
training <- read.csv( url(trainingUrl), header = TRUE, na.strings = c("NA", "#DIV/0!"))
training$user_name <- as.factor(training$user_name); training$cvtd_timestamp <- mdy_hm(training$cvtd_timestamp)
training$new_window <- as.factor(training$new_window); training$num_window <- as.factor(training$num_window)
training$classe <- as.factor(training$classe)
testingUrl <-  "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testing <- read.csv( url(testingUrl), header = TRUE, na.strings = c("NA", "#DIV/0!"))
testing$user_name <- as.factor(testing$user_name); testing$cvtd_timestamp <- mdy_hm(testing$cvtd_timestamp)
testing$new_window <- as.factor(testing$new_window); testing$num_window <- as.factor(testing$num_window)
testing$problem_id <- as.factor(testing$problem_id)
rm(trainingUrl); rm(testingUrl)
```

Subsequently, both data sets were "cleaned" to remove: (a) columns full of NAs, (b) the data set ID number, (c) columns with near zero variation, and (d)
variables that indicated a particular time window because the dynamics to model (dumbbell lifting) are supposed to be independent of time. The same
variables are selected both for the training and testing data sets, with the exception that the testing data set has a variable *problem_id* that is not
present in the training set and that it does not have a defined *classe* (variable to predict).


```r
# Clean data
# Remove columns with NAs
training <- training[, colSums(is.na(training)) == 0] 
training <- subset(training, select = -c(X))
# Examining that the remaining data is variable
nearZero <- nearZeroVar(training, saveMetrics = TRUE)
training <- subset(training, select = names(training[!nearZero$nzv])); rm(nearZero)
# We also know that the system is automonous. Thus, it should not depend on time. We remove time as a feature
training <- subset(training, select = names(training[!grepl("time",names(training))]) );
training <- subset(training, select = -c(num_window))
# Finally, the model is expected to be independent of the subject. We want to make predictions based on observed measurement, not on
# habits of subjects.
training <- subset(training, select = -c(user_name))
# Adjusting testing data set
testing <- subset(testing, select = c(names(training)[1:length(training)-1],"problem_id"))
```

After cleaning the data, we are left with 19622 observations of 52 regressors in the training set, and 
20 data points in the testing set.

A first plot is made to observe the distribution of each

## Data partitioning

The cleaned training set is partitioned into a pure training data set (60%) and a validation data set (40%). The former will be used for training and the
latter for subsequent cross validation.


```r
set.seed(20850)
inTrain <- createDataPartition(training$classe, p = 0.60, list = FALSE)
myTraining <- training[inTrain, ]; myTesting <- training[-inTrain, ]
```

## Machine learning

Random forest, one of the most widely used and accurate prediction models, will be used to automatically classify posture quality. Our choice is supported
by its ability to select important variables while preventing overfitting, commonly observed with decision trees. The method is called with 5-fold cross
validation.

```r
myRandomForest <- train(classe ~ ., data = myTraining, method = "rf",
                        trControl = trainControl(method = "cv", number = 5),
                        prox = TRUE, allowParallel=TRUE)
print(myRandomForest$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.89%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3343    2    1    0    2     0.00149
## B   15 2255    9    0    0     0.01053
## C    0   20 2025    9    0     0.01412
## D    0    3   25 1899    3     0.01606
## E    0    1    6    9 2149     0.00739
```

A random forest is constructed with 500 trees and  27 variables randomly sampled at each
split. The in-sample classification error was found at 0.0160622 in the worst case, proving very accurate.

The obtained model is then used to make predictions in the validation data  set.

```r
prediction <- predict(myRandomForest, myTesting)
cM <- confusionMatrix(myTesting$classe, prediction)
print(cM$overall)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##          0.990          0.988          0.988          0.992          0.285 
## AccuracyPValue  McnemarPValue 
##          0.000            NaN
```

The overall statistics in the validation set show an impressive accuracy: 0.9901861, which is very promising now that we have to make predictions
for the testing set.

## Testing

The proposed random forest model is now used to make prediction in the `testing` data set.

```r
testingPrediction <- predict(myRandomForest, subset(testing, select = -c(problem_id)))
print(testingPrediction)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

These results were tested manually on the Coursera website. All of the testing points were predicted correctly.
