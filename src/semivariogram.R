library(gstat)
library(tidyr)
library(dplyr)
library('spacetime')

# smap_1km
df <- read.csv('data/smap_1km.csv')
df <- df[1:nrow(df), 3:ncol(df)]
df_long <- df %>%
  pivot_longer(cols = -c(coords.x1, coords.x2), names_to = "time", values_to = "value")
df_long$time <- sub("X", "", df_long$time)
df_long$time <- as.Date(df_long$time, format = "%Y%m%d")
df_long = df_long %>% arrange(time)
# standardize value
df_long$value = scale(df_long$value, center = TRUE, scale = TRUE)
data <- df_long

# prcp
df <- read.csv('data/prcp_1km.csv')
df <- df[1:nrow(df), 3:ncol(df)]
df_long <- df %>%
  pivot_longer(cols = -c(coords.x1, coords.x2), names_to = "time", values_to = "value")
df_long$time <- sub("X", "", df_long$time)
df_long$time <- as.Date(df_long$time, format = "%Y%m%d")
df_long = df_long %>% arrange(time)
# standardize value
df_long$value = scale(df_long$value, center = TRUE, scale = TRUE)

data <- data %>%
  rename(smap = value) %>%
  inner_join(df_long %>% rename(prcp= value), by = c("coords.x1", "coords.x2", "time"))

# tmin
df <- read.csv('data/tmin_1km.csv')
df <- df[1:nrow(df), 3:ncol(df)]
df_long <- df %>%
  pivot_longer(cols = -c(coords.x1, coords.x2), names_to = "time", values_to = "value")
df_long$time <- sub("X", "", df_long$time)
df_long$time <- as.Date(df_long$time, format = "%Y%m%d")
df_long = df_long %>% arrange(time)
# standardize value
df_long$value = scale(df_long$value, center = TRUE, scale = TRUE)

data <- data %>%
  inner_join(df_long %>% rename(tmin= value), by = c("coords.x1", "coords.x2", "time"))

# tmax
df <- read.csv('data/tmax_1km.csv')
df <- df[1:nrow(df), 3:ncol(df)]
df_long <- df %>%
  pivot_longer(cols = -c(coords.x1, coords.x2), names_to = "time", values_to = "value")
df_long$time <- sub("X", "", df_long$time)
df_long$time <- as.Date(df_long$time, format = "%Y%m%d")
df_long = df_long %>% arrange(time)
# standardize value
df_long$value = scale(df_long$value, center = TRUE, scale = TRUE)

data <- data %>%
  inner_join(df_long %>% rename(tmax= value), by = c("coords.x1", "coords.x2", "time"))


# srad
df <- read.csv('data/srad_1km.csv')
df <- df[1:nrow(df), 3:ncol(df)]
df_long <- df %>%
  pivot_longer(cols = -c(coords.x1, coords.x2), names_to = "time", values_to = "value")
df_long$time <- sub("X", "", df_long$time)
df_long$time <- as.Date(df_long$time, format = "%Y%m%d")
df_long = df_long %>% arrange(time)
# standardize value
df_long$value = scale(df_long$value, center = TRUE, scale = TRUE)

data <- data %>%
  inner_join(df_long %>% rename(srad = value), by = c("coords.x1", "coords.x2", "time"))



# vp
df <- read.csv('data/vp_1km.csv')
df <- df[1:nrow(df), 3:ncol(df)]
df_long <- df %>%
  pivot_longer(cols = -c(coords.x1, coords.x2), names_to = "time", values_to = "value")
df_long$time <- sub("X", "", df_long$time)
df_long$time <- as.Date(df_long$time, format = "%Y%m%d")
df_long = df_long %>% arrange(time)
# standardize value
df_long$value = scale(df_long$value, center = TRUE, scale = TRUE)

data <- data %>%
  inner_join(df_long %>% rename(vp = value), by = c("coords.x1", "coords.x2", "time"))



spatial_data <- SpatialPoints(data[, c("coords.x1", "coords.x2")] %>% distinct())
proj4string(spatial_data) <- CRS("+proj=utm +zone=15 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
temporal_data <- unique(data$time)


st_data <- STFDF(sp = spatial_data, time = temporal_data, data = data)

vv <- variogram(smap ~ prcp+tmin+tmax+srad+vp+1, st_data, tlags=0, width=1000)

plot(vv)

plot(vv$dist, vv$gamma, type = "l") 



# Assuming `data` is your data matrix or data frame
distance_matrix <- dist(df[, c("coords.x1", "coords.x2")] %>% distinct(), method = "euclidean")
distance_matrix <- as.matrix(distance_matrix)



