library(tidyverse)
library(missForest)
library(caret)
library(FNN)
library(randomForest)
library(e1071)
# UCI veri kümesi - Breast Cancer Wisconsin
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
df <- read_csv(url, col_names = FALSE)
# Veri kümesindeki sütunları isimlendirelim
colnames(df) <- c("id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                  "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
                  "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
                  "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
                  "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                  "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
                  "symmetry_worst", "fractal_dimension_worst")
# 'id' sütununu veri kümesinden çıkaralım
df <- select(df, -id)
# '?' işaretleri yerine NA değerleri kullanalım
df[df == "?"] <- NA
# Eksik verileri knn algoritması ile tamamlayalım
knn_impute <- function(df) {
  knn_model <- knn(train = df, test = df, k = 5, scale = TRUE, use.all = TRUE)
  df[is.na(df)] <- knn_model$imputed[is.na(df)]
  return(df)

}
df_imputed_knn <- knn_impute(df)
# Eksik verileri random forest algoritması ile tamamlayalım
rf_impute <- function(df) {
  rf_model <- randomForest(x = na.roughfix(df), y = NULL, ntree = 500, proximity = TRUE, importance = TRUE)
  df_imputed <- rfImpute(df, proximity = rf_model$proximity, importance = rf_model$importance)
  return(df_imputed)
}
df_imputed_rf <- rf_impute(df)
# Eksik verileri SVM algoritması ile tamamlayalım
svm_impute <- function(df) {
  df_num <- df %>% select_if(is.numeric) # Sayısal sütunları seçelim
  df_num_scaled <- scale(df_num) # Verileri ölçekleyelim
  svm_model <- svm(df_num_scaled, na.action = na.omit) # SVM modeli oluşturalım
  df_imputed <- df_num_scaled # Eksik verileri tamamlanmış veri çerçevesi
  for (i in 1:ncol(df_num)) { # Her sütun için eksik verileri tamamlayalım
    if (sum(is.na(df_num[,i]))) {
      df_num_test <- df_num_scaled