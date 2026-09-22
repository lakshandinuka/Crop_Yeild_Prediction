install.packages("randomForest")
install.packages("gbm")
library(randomForest) 
library(gbm)
library(caret)

df$Season   <- as.factor(df$Season)
df$District <- as.factor(df$District)

set.seed(42)
train_idx <- createDataPartition(df$Paddy_Yield_M.Tonnes, p=0.8, list=FALSE)
train <- df[train_idx, ]; test <- df[-train_idx, ]

rf_model <- randomForest(Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season + District + Year,
                         data=train, ntree=500, importance=TRUE)
rf_pred <- predict(rf_model, test)

gbm_model <- gbm(Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season + District + Year,
                 data=train, distribution="gaussian", n.trees=500, interaction.depth=3)
gbm_pred <- predict(gbm_model, test, n.trees=500)

# Compare R2 / MAE across all models on the same test set
lm_pred <- predict(m1, test)
glm_pred <- predict(m2, test, type="response")

results <- data.frame(
  Model = c("OLS","GLM (Gamma)","Random Forest","GBM"),
  R2  = c(cor(lm_pred, test$Paddy_Yield_M.Tonnes)^2,
          cor(glm_pred, test$Paddy_Yield_M.Tonnes)^2,
          cor(rf_pred, test$Paddy_Yield_M.Tonnes)^2,
          cor(gbm_pred, test$Paddy_Yield_M.Tonnes)^2),
  MAE = c(mean(abs(lm_pred - test$Paddy_Yield_M.Tonnes)),
          mean(abs(glm_pred - test$Paddy_Yield_M.Tonnes)),
          mean(abs(rf_pred - test$Paddy_Yield_M.Tonnes)),
          mean(abs(gbm_pred - test$Paddy_Yield_M.Tonnes))))
results

importance(rf_model)          # which predictors matter most — compare against your LASSO coefficients
varImpPlot(rf_model)