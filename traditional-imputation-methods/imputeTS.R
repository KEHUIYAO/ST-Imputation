library(imputeTS)
library(dplyr)
library(tidyr)
library(data.table)

impute_column_with_fallback <- function(column_data) {
  result <- tryCatch(
    {
      warning_status <- FALSE
      imputed_data <- withCallingHandlers(
        na_kalman(column_data),
        warning = function(w) {
          warning_status <<- TRUE
          invokeRestart("muffleWarning")
        }
      )
      
      if (warning_status) {
        imputed_data <- na_interpolation(column_data)
      }
      imputed_data
    },
    error = function(e) {
      imputed_data <- na_interpolation(column_data)
      imputed_data
    }
  )
  return(result)
}


df <- read.csv('../data/SMAP_Climate_In_Situ_TxSON.csv')

# Convert to Date
df$Date <- as.Date(as.character(df$Date), format='%Y%m%d')

# Filter data for the year 2017
y <- df %>% filter(format(Date, "%Y") == "2017")

setDT(y)
y <- dcast(y, Date ~ POINTID, value.var = "SMAP_1km")[, -"Date", with = FALSE]
y <- as.matrix(y)


# Randomly mask out rows with probability p = 0.2
p <- 0.2
n_rows <- nrow(y)
time_points_to_eval <- sample(n_rows, size = round(p * n_rows), replace = FALSE)

eval_mask <- matrix(0, nrow = n_rows, ncol = ncol(y))
eval_mask[time_points_to_eval, ] <- 1

observed_mask <- matrix(1, nrow = n_rows, ncol = ncol(y))
observed_mask[is.na(y)] <- 0
eval_mask <- eval_mask * observed_mask
training_mask <- observed_mask - eval_mask

y_train <- y
y_train[training_mask == 0] <- NA
y_val <- y
y_val[eval_mask == 0] <- NA

y_imputed <- y_train
for (i in 1:ncol(y_train)) {
  y_imputed[,i] <- impute_column_with_fallback(y_train[,i])
}



y_imputed_eval <- y_imputed[eval_mask == 1]
y_val_eval <- y_val[eval_mask == 1]
mae <- mean(abs(y_imputed_eval - y_val_eval))
print(mae)







