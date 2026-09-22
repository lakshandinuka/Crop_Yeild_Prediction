install.packages(c("tidyverse","readxl","skimr","DataExplorer","car","lmtest","MASS","glmnet","caret","agricolae","forecast","BAS"))
library(tidyverse)
library(readxl)

df <- read_excel("/Users/dinuka/SLIIT/Y3S1/SM/Project/Crop_Yeild_Prediction/Dataset/rainfall_agricultural_productivity.xlsx")

df$District <- as.factor(df$District)
df$Season <- as.factor(df$Season)

skimr::skim(df)
DataExplorer::plot_missing(df)

# Descriptive stats
df %>% group_by(Season) %>%
  summarise(mean_rain = mean(Seasonal_Rainfall_mm), 
            sd_rain = sd(Seasonal_Rainfall_mm),
            mean_yield = mean(Paddy_Yield_M.Tonnes), 
            sd_yield = sd(Paddy_Yield_M.Tonnes))

df %>% group_by(District) %>%
  summarise(mean_rain = mean(Seasonal_Rainfall_mm), 
            mean_yield = mean(Paddy_Yield_M.Tonnes)) %>%
  arrange(desc(mean_yield))

# Outliers
boxplot(df$Seasonal_Rainfall_mm ~ df$Season)
ggplot(df, aes(x=Seasonal_Rainfall_mm, y=Paddy_Yield_M.Tonnes)) + geom_point(aes(color=Season)) + geom_smooth(method="lm") + theme_minimal()

ggplot(df, aes(x=District, y=Seasonal_Rainfall_mm)) + geom_boxplot() + coord_flip()
ggplot(df, aes(x=Year, y=Paddy_Yield_M.Tonnes)) + geom_line(aes(group=District), alpha=0.3) + stat_summary(fun=mean, geom="line", color="red", size=1.2)

