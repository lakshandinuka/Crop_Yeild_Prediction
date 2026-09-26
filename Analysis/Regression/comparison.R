# Compare R2 / MAE across all models on the same test set
lm_pred <- predict(m1, test)
glm_pred <- predict(m2, test, type="response")

actual <- test$Paddy_Yield_M.Tonnes

R2 <- c(
  1 - sum((test$Paddy_Yield_M.Tonnes - lm_pred)^2) /
    sum((test$Paddy_Yield_M.Tonnes - mean(test$Paddy_Yield_M.Tonnes))^2),
  
  1 - sum((test$Paddy_Yield_M.Tonnes - glm_pred)^2) /
    sum((test$Paddy_Yield_M.Tonnes - mean(test$Paddy_Yield_M.Tonnes))^2),
  
  1 - sum((test$Paddy_Yield_M.Tonnes - rf_pred)^2) /
    sum((test$Paddy_Yield_M.Tonnes - mean(test$Paddy_Yield_M.Tonnes))^2),
  
  1 - sum((test$Paddy_Yield_M.Tonnes - gbm_pred)^2) /
    sum((test$Paddy_Yield_M.Tonnes - mean(test$Paddy_Yield_M.Tonnes))^2)
)

results <- data.frame(
  Model = c("OLS","GLM (Gamma)","Random Forest","GBM"),
  
  R2 = as.numeric(R2),
  
  MAE = c(mean(abs(lm_pred - test$Paddy_Yield_M.Tonnes)),
          mean(abs(glm_pred - test$Paddy_Yield_M.Tonnes)),
          mean(abs(rf_pred - test$Paddy_Yield_M.Tonnes)),
          mean(abs(gbm_pred - test$Paddy_Yield_M.Tonnes))))
results

importance(rf_model)          # which predictors matter most — compare against your LASSO coefficients
varImpPlot(rf_model)