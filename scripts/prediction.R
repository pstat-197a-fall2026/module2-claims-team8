library(tidyverse)
library(keras3)
source('scripts/preprocessing-primary.R')
load('data/claims-test.RData')
load('data/claims-raw.RData')   # only used for class labels

# gives better binary accuracy than 0.825
USE_MCLASS_FOR_BINARY <- FALSE

# class labels, in the order the models were trained on
bclass_labels <- levels(claims_raw$bclass)   # "N/A: No relevant content." / "Relevant claim content"
mclass_labels <- levels(claims_raw$mclass)

## PREPROCESS -------------------------------
clean_df <- claims_test %>%
  parse_data() %>%
  select(.id, text_clean)
stopifnot(nrow(clean_df) == 929)   # stops if any page got dropped

x <- clean_df %>% pull(text_clean)

## MULTICLASS -------------------------------
m_model <- load_model('results/mclass-nn-model.keras')
m_probs <- predict(m_model, x)                  # one column per class
mclass.pred <- factor(mclass_labels[max.col(m_probs)],
                      levels = mclass_labels)

## BINARY -----------------------------------
if (USE_MCLASS_FOR_BINARY) {
  # "N/A" multiclass prediction -> "N/A" binary, anything else -> relevant
  bclass.pred <- factor(if_else(mclass.pred == mclass_labels[1],
                                bclass_labels[1], bclass_labels[2]),
                        levels = bclass_labels)
} else {
  b_model <- load_model('results/bclass-nn-model.keras')
  b_probs <- predict(b_model, x) %>% as.numeric()
  bclass.pred <- factor(if_else(b_probs > 0.5, bclass_labels[2], bclass_labels[1]),
                        levels = bclass_labels)
}

## EMPTY PAGES ------------------------------
# pages with no usable text (PDFs, errors) -> "no relevant content"
empty <- clean_df$text_clean == ''
bclass.pred[empty] <- bclass_labels[1]
mclass.pred[empty] <- mclass_labels[1]

## EXPORT (same format as example) ----------
pred_df <- clean_df %>%
  select(.id) %>%
  mutate(bclass.pred = bclass.pred,
         mclass.pred = mclass.pred)

# final checks
nrow(pred_df)               # should be 929
anyNA(pred_df)              # should be FALSE
table(pred_df$bclass.pred)
table(pred_df$mclass.pred)
sum(empty)                  # number of empty pages

save(pred_df, file = 'results/preds-group8.RData')