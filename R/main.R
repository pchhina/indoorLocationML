mpg <- mpg
str(mpg)
na_vec <- apply(mpg, 2, function(x) mean(is.na(x)) > c(0,0.5))
str(na_vec)
sum(na_vec)

rm(list = ls())
library(tidyverse)
library(caret)
library(tictoc)
library(corrplot)
library(e1071)
src <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip"

setwd("../inst/extdata")

if (!file.exists("UJIndoorLoc.zip")) {
    download.file(url = src,
                  destfile = "UJIndoorLoc.zip")
}

unzip("UJIndoorLoc.zip")
trainingData <- read_csv("UJIndoorLoc/trainingData.csv")

# Let's focus first on longitude prediction

trainLong <- trainingData[, 1:521]
trX <- trainLong[, -521]
trY <- trainLong[, 521]
# Check if any vector has NAs
sum(apply(trX, 2, function(x) sum(is.na(x)))) != 0

# Analysis of Zero Variance
nzv_vec <- nearZeroVar(trX, saveMetrics = TRUE)
sum(nzv_vec$zeroVar)
trX <- trX[, !nzv_vec$zeroVar] # remove zero variance predictors

# Find skewness
skew_vec <- apply(trX, 2, skewness)
sum(abs(skew_vec) > 1) # Many vectors are skewed because of 100 value for no signal

# Analysis of highly correlated variables
cormat <- cor(trX)
corrplot(cormat, order = "hclust") # not many
corrTh <- 0.9
tooHigh <- findCorrelation(cor(trX), corrTh)
trX <- trX[, -tooHigh] # remove predictors with more than 0.9 correlation
dim(trX)

# Is scaling a problem with the dataset?
hist(apply(trX, 2, function(x) max(x) - min(x))) # scales of predictors is not an issue

# Let's still transform, it doesn't hurt
pp <- preProcess(trX, method = c("BoxCox", "center", "scale"))
print(pp)
pp.trX <- predict(pp, trX)

# Let's do PCA
pca_comp <- prcomp(trX)
var.x <- pca_comp$sdev^2
pvar.x <- var.x / sum(var.x)
plot(cumsum(pvar.x), type = "b") # may be helpful as alternate set to fit

control <- trainControl(method = "cv",
                        number = 5,
                        verboseIter = TRUE)
training <- cbind(trX, trY)
# Let's try the knn model
tic("knn")
set.seed(521)
knnFit <- train(LONGITUDE ~ .,
               data = training,
               method = "knn",
               trControl = control)
toc()

# Let's try GBM
tic("gbm")
set.seed(521)
gbmGrid = expand.grid(interaction.depth = seq(1, 7, by = 2),
                      n.trees = seq(100, 1000, by = 50),
                      shrinkage = c(0.01, 0.1),
                      n.minobsinnode = 10)
gbmFit <- train(LONGITUDE ~ .,
               data = training,
               method = "gbm",
               tuneGrid = gbmGrid,
               distribution = "gaussian",
               trControl = control)
toc()

# rgression equations may effect svm method so may be try at the end 
# let's try random forest
tic("rf")
set.seed(521)
rfFit <- train(LONGITUDE ~ .,
               data = training,
               method = "rf",
               trControl = control)
toc()


# Comparing the models
resamps <- list(KNN = knnFit, GBM = gbmFit, RF = rfFit)
resamps <- resamples(resamps)
dotplot(resamps)
testSamples <- createDataPartition(training$LONGITUDE, p = 0.1, list = FALSE)
xyplot(testSamples$LONGITUDE ~ predict(rfFit, newdata = testSamples))
xyplot(resid(rfFit, newdata = testSamples) ~ predict(rfFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(knnFit, newdata = testSamples))
xyplot(resid(knnFit, newdata = testSamples) ~ predict(knnFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(gbmFit, newdata = testSamples))
xyplot(resid(gbmFit, newdata = testSamples) ~ predict(gbmFit, newdata = testSamples))
