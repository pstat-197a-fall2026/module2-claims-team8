#Build and train a neural network model. 
#Experiment with architectures and training configurations until you find 
#a model that performs at least as well as principal component logistic regression from task 1. 
#Quantify the predictive accuracy.

library(keras3)
source('scripts/preprocessing-primary.R')

library(tensorflow)
tf$constant('Hello world')
library(tidyverse)
library(tidymodels)
library(tidytext)

load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()


set.seed(111225)
partitions <- claims_clean %>%
  mutate(text_clean = str_trim(text_clean)) %>%
  filter(str_length(text_clean) > 5) %>%
  initial_split(prop = 0.8)


train_dtm <- training(partitions) %>%
  unnest_tokens(output = 'token', 
                input = text_clean) %>%
  group_by(.id, bclass) %>%
  count(token) %>%
  bind_tf_idf(term = token, 
              document = .id, 
              n = n) %>%
  pivot_wider(id_cols = c(.id, bclass), 
              names_from = token, 
              values_from = tf_idf,
              values_fill = 0) %>%
  ungroup()

x_train <- train_dtm %>%
  select(-bclass, -.id) %>%
  as.matrix()

storage.mode(x_train) <- "numeric"

# extract labels and coerce to binary
y_train <- train_dtm %>% 
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1


#model

model_task3 <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')


model_task3 %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = tf$keras$optimizers$SGD(),
    metrics = 'binary_accuracy')

#50 epochs
history_task3 <- model_task3 %>%
  fit(x = x_train,
      y = y_train,
      epochs = 50)

plot(history_task3)

# new model
model_task3_2 <- model_task3 %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'binary_accuracy')

# re-train
history_task3_2 <- model_task3_2 %>%
  fit(x = x_train,
      y = y_train,
      epochs = 10)

plot(history_task3_2)

# 3

model_task3_3 <- model_task3 %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'binary_accuracy')

# train with validation split
history_task3_3 <- model_task3_3 %>%
  fit(x = x_train,
      y = y_train,
      epochs = 20,
      validation_split = 0.2)


plot(history_task3_3)
