##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#first look on data
str(edx)

#splitting into train and test set
test_index <- createDataPartition(y=edx$rating, times=1, p=0.1, list=FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

#distinct movies
tibble(unique_genres=nrow(distinct(edx, movieId))) %>% knitr::kable()

#ratings per movie
edx %>%
  group_by(movieId) %>%
  summarize(ratings_per_movie=n()) %>%
  ggplot(aes(ratings_per_movie)) +
  geom_histogram(color="black", bins = 25) +
  scale_x_log10()

#distinct users
tibble(unique_genres=nrow(distinct(edx, userId))) %>% knitr::kable()

#ratings per user
edx %>%
  group_by(userId) %>%
  summarize(ratings_per_user=n()) %>%
  ggplot(aes(ratings_per_user)) +
  geom_histogram(color="black", bins=25) +
  scale_x_log10()

#distinct genres
tibble(unique_genres=nrow(distinct(edx, genres))) %>% knitr::kable()

#ratings per genres
edx %>%
  group_by(genres) %>%
  summarize(ratings_per_genres=n()) %>%
  ggplot(aes(ratings_per_genres)) +
  geom_histogram(color="black", bins=25) +
  scale_x_log10()

#mean user ratings
edx %>%
  group_by(userId) %>%
  summarize(mean_user_ratings=mean(rating)) %>%
  ggplot(aes(mean_user_ratings)) +
  geom_histogram(color="black", bins=25)

#RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  s <- sqrt(mean((true_ratings - predicted_ratings)^2, na.rm=TRUE))
  s
}

#1. first model (predict same rating for all movies and users)
#naive rmse
mu <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse
tibble(method="naive rmse", 
       RMSE=naive_rmse) %>% 
  knitr::kable()

#2. Modeling Movie effects
#average rating per movie
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i=mean(rating - mu))

#Prediction for test_set
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

#shows the variation from mean (mean is ~3.512)
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

#RMSE for test_set
rmse_movie_effect <- RMSE(test_set$rating, predicted_ratings)
tibble(method=c("naive rmse",
                "movie effect"), 
       RMSE=c(naive_rmse, 
              rmse_movie_effect)) %>% 
  knitr::kable()

#3. user effects
#different users
train_set %>%
  distinct(userId) %>%
  summarize(n())

#average rating for users rated over 100 movies (my version)
train_set %>%
  group_by(userId) %>%
  summarize(b_u=mean(rating), n=n()) %>%
  filter(n>100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#average rating per user
user_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=mean(rating-mu-b_i))

#Prediction and RMSE for test_set
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
rmse_user_effect <- RMSE(test_set$rating, predicted_ratings)
rmse_user_effect
tibble(method=c("naive rmse",
                "movie effect",
                "user effect"),
       RMSE=c(naive_rmse, 
              rmse_movie_effect,
              rmse_user_effect)) %>%
  knitr::kable()

#4. genre effects
#count of different genres
train_set %>%
  filter(genres!="(no genres listed)") %>%
  distinct(genres) %>%
  summarize(n())

#show movies per genre
train_set %>%
  filter(genres!="(no genres listed)") %>%
  group_by(genres) %>%
  summarize(movies_per_genre=n()) %>%
  ggplot(aes(x=movies_per_genre, y=..count..)) +
  geom_histogram(stat="bin", bins = 20, color="black") +
  scale_x_continuous(trans="log2")

#rating per genre (with +10 movies)
train_set %>%
  filter(genres!="(no genres listed)") %>%
  group_by(genres) %>%
  summarize(b_g=mean(rating), n=n()) %>%
  filter(n>=10) %>%
  arrange(b_g) %>%
  ggplot(aes(b_g)) +
  geom_histogram(bins = 30, color = "black") +
  labs(title="average rating per genre", 
       x="rating per genre", 
       y="count")

#genresaverages
genres_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g=mean(rating-mu-b_i-b_u))

#Prediction and RMSE for test_set
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(genres_avgs, by="genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
rmse_genres_effect <- RMSE(test_set$rating, predicted_ratings)
tibble(method=c("naive rmse",
                "movie effect", 
                "user effect", 
                "genres effect"), 
       RMSE=c(naive_rmse,
              rmse_movie_effect,
              rmse_user_effect,
              rmse_genres_effect)) %>% 
  knitr::kable()

#Regularization
#Penalized least squares

#sequence of lambdas
lambdas <- seq(0, 10, 0.25)

#Takes much time
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  #movie effect with regularization on train set
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #user effect with regularization on train set
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #genres effect with regularization on train set
  b_g <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  #predictions on test set
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

#plotting the lambdas vs rmses
qplot(lambdas, rmses)

#getting minimum lambda
lambda <- lambdas[which.min(rmses)]

#minimum rmse
regularization_train_test <- min(rmses)

#finally regularize with edx and validation set
#mean of edx ratings
mu <- mean(edx$rating)

#movie effect with regularization on edx set
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

#user effect with regularization on edx set
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

#genres effect with regularization on edx set
b_g <- edx %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))

#predictions on validation set
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

regularization_edx_validation <- RMSE(predicted_ratings, validation$rating)

#Final overview of all modelings and final prediction of regularization on validation set
tibble(method=c("naive rmse",
               "movie effect", 
               "user effect", 
               "genres effect",
               "regularized movie, user and genres effect (train and test)",
               "regularized movie, user and genres effect (edx and validation)"), 
      RMSE=c(naive_rmse,
             rmse_movie_effect,
             rmse_user_effect,
             rmse_genres_effect,
             regularization_train_test,
             regularization_edx_validation)) %>% 
  knitr::kable()