# Convert categorical variables to factors
df$Season   <- as.factor(df$Season)
df$District <- as.factor(df$District)

t.test(Seasonal_Rainfall_mm ~ Season, data = df)

t.test(Paddy_Yield_M.Tonnes ~ Season, data = df)

model_aov <- aov(Seasonal_Rainfall_mm ~ District * Season, data = df)
summary(model_aov)
TukeyHSD(model_aov)   # which districts differ pairwise

bartlett.test(Seasonal_Rainfall_mm ~ Season, data = df)  # or var.test() for 2 groups

# Create low-yield indicator
df$low_yield <- ifelse(
  df$Paddy_Yield_M.Tonnes < median(df$Paddy_Yield_M.Tonnes, na.rm = TRUE), 1, 0)

# Descriptive proportions by season
prop_table <- prop.table(table(df$Season, df$low_yield), margin = 1)
prop_table

# Create contingency table
tab <- table(df$Season, df$low_yield)

# Two-proportion test
prop_test <- prop.test(x = c(tab["Maha", "1"], tab["Yala", "1"]),
  n = c(
    sum(df$Season == "Maha"), sum(df$Season == "Yala")), correct = FALSE)

prop_test
