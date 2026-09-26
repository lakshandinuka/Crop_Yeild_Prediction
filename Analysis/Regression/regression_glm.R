library(caret)
library(randomForest)
library(gbm)
library(glmnet)

# Make sure data is ordered chronologically
df <- df[order(df$Year), ]

# ============================================================
# 1. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# Use the earlier years for training and the latest years for testing
# Example: last 20% of unique years are used as test years

years <- sort(unique(df$Year))

n_test_years <- ceiling(length(years) * 0.20)

test_years <- tail(years, n_test_years)
train_years <- head(years, length(years) - n_test_years)

train <- df[df$Year %in% train_years, ]
test  <- df[df$Year %in% test_years, ]

cat("Training years:", min(train$Year), "-", max(train$Year), "\n")
cat("Testing years :", min(test$Year), "-", max(test$Year), "\n")

# Check the number of observations
cat("Training observations:", nrow(train), "\n")
cat("Testing observations :", nrow(test), "\n")


# ============================================================
# 2. MULTIPLE LINEAR REGRESSION (OLS)
# ============================================================

m1 <- lm(
  Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season +
    District + Year,
  data = train
)

summary(m1)

# Residual diagnostics
par(mfrow = c(2, 2))
plot(m1)

# Multicollinearity
car::vif(m1)


# ============================================================
# 3. GLM
# ============================================================

m2 <- glm(
  Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season +
    District + Year,
  family = Gamma(link = "log"),
  data = train
)

summary(m2)


# ============================================================
# 4. OLS AND GLM PREDICTIONS ON UNSEEN TEST DATA
# ============================================================

lm_pred <- predict(m1, newdata = test)

glm_pred <- predict(
  m2,
  newdata = test,
  type = "response"
)


# ============================================================
# 5. RIDGE / LASSO / ELASTIC NET
# ============================================================

# Create model matrix USING TRAINING DATA
X_train <- model.matrix(
  Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm +
    Season + District + Year,
  data = train
)[, -1]

y_train <- train$Paddy_Yield_M.Tonnes

# Create test matrix using the same predictors
X_test <- model.matrix(
  Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm +
    Season + District + Year,
  data = test
)[, -1]

# LASSO
set.seed(42)

cv_lasso <- cv.glmnet(
  X_train,
  y_train,
  alpha = 1
)

coef(cv_lasso, s = "lambda.min")

# LASSO predictions
lasso_pred <- predict(
  cv_lasso,
  newx = X_test,
  s = "lambda.min"
)


# ============================================================
# 6. STEPWISE SELECTION
# ============================================================

step_model <- step(
  lm(
    Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm +
      Season + District + Year,
    data = train
  ),
  direction = "both"
)

summary(step_model)

# Stepwise model prediction
step_pred <- predict(
  step_model,
  newdata = test
)


# ============================================================
# 7. RANDOM FOREST
# ============================================================

set.seed(42)

rf_model <- randomForest(
  Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm +
    Season + District + Year,
  data = train,
  ntree = 500,
  importance = TRUE
)

rf_pred <- predict(
  rf_model,
  newdata = test
)


# ============================================================
# 8. GRADIENT BOOSTING MACHINE
# ============================================================

set.seed(42)

gbm_model <- gbm(
  Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm +
    Season + District + Year,
  data = train,
  distribution = "gaussian",
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  verbose = FALSE
)

gbm_pred <- predict(
  gbm_model,
  newdata = test,
  n.trees = 500
)
