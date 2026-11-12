
# Load libraries and Data
library(tidyverse)
library(keras3)
library(tidymodels)

keras3::set_random_seed(2026)

load("data/claims-clean-example.RData")

claims <- claims_clean |>
  mutate(text_clean = replace_na(text_clean, ""),
         bclass = factor(bclass),
         mclass = factor(mclass))

levels(claims$mclass)
levels(claims$bclass)

# Train/Validation split

set.seed(2026)

claims_split <- initial_split(claims, prop = 0.8, strata = bclass)

train <- training(claims_split)
val <- testing(claims_split)

saveRDS(val$.id, "data/val-ids.rds")

nrow(train)
nrow(val)

# Section 3: TF-IDF vectorizer
vectorizer <- layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  max_tokens = 10000,
  output_mode = "tf_idf"
)

adapt(vectorizer, train$text_clean)

# check: how many words did it learn?
length(get_vocabulary((vectorizer)))

# check: what does one page look like as numbers?
dim(as.array(vectorizer(train$text_clean[1:3])))

# Section 4: Binary Model (relevant vs. not relevant)

# labels as 0/1
y_train_b <- as.numeric(train$bclass == "Relevant claim content")
y_val_b <- as.numeric(val$bclass == "Relevant claim content")

model_b <- keras_model_sequential() |>
  vectorizer() |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 1, activation = "sigmoid")

# training settings
model_b |> compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy",
)

# train
history_b <- model_b |> fit(
  train$text_clean, y_train_b,
  validation_data = list(val$text_clean, y_val_b),
  epochs = 30,
  batch_size = 32,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5,
                            restore_best_weights = TRUE)
  )
)

plot(history_b)

# accuracy on validation pages
eval_b <- model_b |> evaluate(val$text_clean, y_val_b)
eval_b

# which pages did it get right / wrong?
prob_b <- predict(model_b, val$text_clean)
pred_b <- ifelse(prob_b > 0.5, 1, 0)

table(predicted = pred_b, actual = y_val_b)

# Section 5: Multiclass model (5 categories)

class_names <- levels(claims$mclass)
n_classes <- length(class_names)

# labels as 0, 1, 2, 3, 4
y_train_m <- as.integer(train$mclass) - 1
y_val_m <- as.integer(val$mclass) - 1

# free memory from the binary model (so RStudio doesnt freeze)
keras3::clear_session()

model_m <- keras_model_sequential() |>
  vectorizer() |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = n_classes, activation = "softmax")

# training settings
model_m |> compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# train
history_m <- model_m |> fit(
  train$text_clean, y_train_m,
  validation_data = list(val$text_clean, y_val_m),
  epochs = 30,
  batch_size = 32,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5,
                            restore_best_weights = TRUE)
  )
)

plot(history_m)

# accuracy on validation page
eval_m <- model_m |>
  evaluate(val$text_clean, y_val_m)
eval_m

# which categories did it get right / wrong?
prob_m <- predict(model_m, val$text_clean)
pred_m <- class_names[max.col(prob_m)]

table(predicted = pred_m, actual = val$mclass)

# ------------------------
# Section 6: Tuning
# ------------------------



