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

# nnets may also suffer because of linear equations used but let's try
# turned out to be worse of all, long tuning time, worst performance!!
nnGrid <- expand.grid(decay = c(0, 0.01, 0.1),
                      size = c(1:10),
                      bag = FALSE)
tic("nn")
set.seed(521)
nnFit <- train(LONGITUDE ~ .,
               data = training,
               method = "avNNet",
               tuneGrid = nnGrid,
               trControl = control,
               linout = TRUE,
               trace = FALSE,
               MaxNWts = 10 * (ncol(training)) + 10 + 1,
               maxit = 500)
toc()

# since svm with radial kernel also has a local behavior, let's try that
# performed poorly, 12 hrs to tune and 0.26 best Rsquared
tic("svm")
set.seed(521)
svmrFit <- train(LONGITUDE ~ .,
                 data = training,
                 method = "svmRadial",
                 tuneLength = 14,
                 trControl = control)
toc()

# let's try MARS. It may not work due to regression approach
# not that great
tic("mars")
set.seed(521)
marsFit <- train(LONGITUDE ~ .,
                 data = training,
                 method = "earth",
                 tuneGrid = data.frame(nprune = seq(2, 450, by = 5),
                                       degree = 1),
                 trControl = control)
toc()

# Cubist is also a linear fit approach so may not work well. Let's try.
# well this worked surprisingly well!
# this is again why you should try different methods
tic("cubist")
set.seed(521)
cubistFit <- train(LONGITUDE ~ .,
                   data = training,
                   method = "cubist",
                   trControl = control)
toc()

# since PLS is fundamentally a linear regression, not sure if that will work
# but still worth a try, other models have surprised too!
# surprisingly fast(under a minute!) with 94% R-squared!
tic("pls")
set.seed(521)
plsFit <- train(LONGITUDE ~.,
                data = training,
                method = "pls",
                tuneLength = 50,
                trControl = control)
toc()

# Let's try M5 model tree
tic("m5")
set.seed(521)
m5Fit <- train(LONGITUDE ~ .,
               data = training,
               method = "M5",
               trControl = control)
toc()

# Comparing the models
resamps <- list(KNN = knnFit, 
                GBM = gbmFit, 
                RF = rfFit, 
                NN = nnFit, 
                SVM = svmrFit,
                MARS = marsFit, 
                CUBIST = cubistFit,
                PLS = plsFit)
resamps <- resamples(resamps)
dotplot(resamps)
dotplot(resamps, metric = "Rsquared")

testSamples <- createDataPartition(training$LONGITUDE, p = 0.1, list = FALSE)
xyplot(testSamples$LONGITUDE ~ predict(rfFit, newdata = testSamples))
xyplot(resid(rfFit, newdata = testSamples) ~ predict(rfFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(knnFit, newdata = testSamples))
xyplot(resid(knnFit, newdata = testSamples) ~ predict(knnFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(gbmFit, newdata = testSamples))
xyplot(resid(gbmFit, newdata = testSamples) ~ predict(gbmFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(nnFit, newdata = testSamples))
xyplot(resid(nnFit, newdata = testSamples) ~ predict(nnFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(svmrFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(marsFit, newdata = testSamples))
xyplot(resid(marsFit, newdata = testSamples) ~ predict(marsFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(cubistFit, newdata = testSamples))
xyplot(resid(cubistFit, newdata = testSamples) ~ predict(cubistFit, newdata = testSamples))

xyplot(testSamples$LONGITUDE ~ predict(plsFit, newdata = testSamples))
xyplot(resid(plsFit, newdata = testSamples) ~ predict(plsFit, newdata = testSamples))
