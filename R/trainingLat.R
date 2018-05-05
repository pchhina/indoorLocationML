rm(list = ls())
library(tidyverse)
library(caret)
library(tictoc)
library(corrplot)
library(e1071)

trainingData <- read_csv("../inst/extdata/UJIndoorLoc/trainingData.csv")

# Now we will try top 5 algorithms on Latitude

trYLat <- trainingData[, 522]
trX <- training[, -448]
training <- cbind(trX, trYLat)

# Random Forest
tic("rf")
set.seed(521)
rfFitLat <- train(LATITUDE ~ .,
               data = training,
               method = "rf",
               trControl = control)
timeRF <- toc()

# Cubist
tic("cubist")
set.seed(521)
cubistFitLat <- train(LATITUDE ~ .,
                   data = training,
                   method = "cubist",
                   trControl = control)
timeCubist <- toc()

#knn
tic("knn")
set.seed(521)
knnFitLat <- train(LATITUDE ~ .,
                   data = training,
                   method = knn,
                   trControl = control)
timeKnn <- toc()

# GBM
tic("gbm")
set.seed(521)
gbmGrid = expand.grid(interaction.depth = seq(1, 7, by = 2),
                      n.trees = seq(100, 1000, by = 50),
                      shrinkage = c(0.01, 0.1),
                      n.minobsinnode = 10)
gbmFitLat <- train(LATITUDE ~ .,
               data = training,
               method = "gbm",
               tuneGrid = gbmGrid,
               distribution = "gaussian",
               trControl = control)
timeGBM <- toc()

# PLS
tic("pls")
set.seed(521)
plsFitLat <- train(LATITUDE ~.,
                data = training,
                method = "pls",
                tuneLength = 50,
                trControl = control)
timePLS <- toc()


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

testSamples <- createDataPartition(training$LATITUDE, p = 0.1, list = FALSE)
testSamples <- training[testSamples,]
xyplot(testSamples$LATITUDE ~ predict(rfFitLat, newdata = testSamples))
xyplot(resid(rfFitLat, newdata = testSamples) ~ predict(rfFitLat, newdata = testSamples))

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
