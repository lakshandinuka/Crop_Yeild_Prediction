wide <- df %>% select(District, Year, Season, Seasonal_Rainfall_mm) %>%
  unite(period, Year, Season) %>% pivot_wider(names_from=period, values_from=Seasonal_Rainfall_mm)


library(missMDA)

# Perform PCA with missing values
res_pca <- imputePCA(wide %>% select(-District), ncp = 2)
pca <- prcomp(res_pca$completeObs, scale. = TRUE)

summary(pca); biplot(pca)