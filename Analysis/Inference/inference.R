t.test(Seasonal_Rainfall_mm ~ Season, data = df)

t.test(Paddy_Yield_M.Tonnes ~ Season, data = df)

model_aov <- aov(Seasonal_Rainfall_mm ~ District, data = df)
summary(model_aov)
TukeyHSD(model_aov)   # which districts differ pairwise

bartlett.test(Seasonal_Rainfall_mm ~ Season, data = df)  # or var.test() for 2 groups

df$low_yield <- ifelse(df$Paddy_Yield_M.Tonnes < median(df$Paddy_Yield_M.Tonnes), 1, 0)
prop.table(table(df$Season, df$low_yield), margin=1)
