# Multiple linear regression
df$Log_Paddy_Production <- log(df$Paddy_Yield_M.Tonnes)
m1 <- lm(Log_Paddy_Production ~ Seasonal_Rainfall_mm + Season + District + Year, data=df)
summary(m1)
par(mfrow=c(2,2)); plot(m1)     # residual diagnostics
car::vif(m1)                     # multicollinearity check

# GLM (e.g. Gamma with log link, since yield is positive/continuous)
m2 <- glm(Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season + District,
          family = Gamma(link="log"), data=df)
summary(m2)

# Ridge / LASSO / Elastic Net
library(glmnet)
X <- model.matrix(Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season + District + Year, df)[,-1]
y <- df$Paddy_Yield_M.Tonnes
cv_lasso <- cv.glmnet(X, y, alpha=1)   # alpha=1 LASSO, 0 Ridge, 0.5 Elastic Net
coef(cv_lasso, s="lambda.min")

# Stepwise selection
step(lm(Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season + District + Year, data=df),
     direction="both")



