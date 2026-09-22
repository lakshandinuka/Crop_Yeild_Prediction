library(BAS)
bayes_model <- bas.lm(Paddy_Yield_M.Tonnes ~ Seasonal_Rainfall_mm + Season + District,
                      data=df, prior="ZS-null")
summary(bayes_model)
coef(bayes_model)