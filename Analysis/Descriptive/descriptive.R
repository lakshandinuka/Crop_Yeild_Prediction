install.packages(c("tidyverse","readxl","skimr","DataExplorer","car","lmtest","MASS","glmnet","caret","agricolae","forecast","BAS"))
library(tidyverse)
library(readxl)

df <- read_excel("/Users/dinuka/SLIIT/Y3S1/SM/Project/Crop_Yeild_Prediction/Dataset/rainfall_agricultural_productivity.xlsx")

df$District <- as.factor(df$District)
df$Season <- as.factor(df$Season)

skimr::skim(df)