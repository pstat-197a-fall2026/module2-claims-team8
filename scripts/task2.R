## TASK 2: do bigrams capture additional information about claims status?
##
## Stage 1: logistic PCR on word tokens -> predicted log-odds
## Stage 2: logistic regression on [stage-1 log-odds + top PCs of bigram tokens]
## Compare stage-2 test accuracy with the word-only (stage-1) model.
##
source('task1_preprocessing.R')
suppressPackageStartupMessages(library(tidymodels))

find_file <- function(f){
  for(d in c('data', '.')){
    p <- file.path(d, f)
    if(file.exists(p)) return(p)
  }
  stop('cannot find ', f)
}

load(find_file('claims-raw.RData'))   # creates claims_raw
lev <- levels(factor(claims_raw$bclass))

## 1. PREPROCESS: word tokens and bigram tokens ---------------------------------
# bigrams of the cleaned text (stopwords removed before forming bigrams)
task2_bigram_fn <- function(parse_data.out, min_df = 10){
  parse_data.out %>%
    unnest_tokens(output = bigram,
                  input = text_clean,
                  token = 'ngrams',
                  n = 2,
                  stopwords = str_remove_all(stop_words$word, '[[:punct:]]')) %>%
    filter(!is.na(bigram)) %>%
    count(.id, bclass, bigram, name = 'n') %>%
    group_by(bigram) %>%
    filter(n_distinct(.id) >= min_df) %>%
    ungroup() %>%
    bind_tf_idf(term = bigram, document = .id, n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'bigram',
                values_from = 'tf_idf',
                values_fill = 0)
}

cache <- 'task2-cache.RData'
if(file.exists(cache)){
  load(cache)
}else{
  cat('parsing html...\n')
  parsed <- claims_raw %>% task1_parse_data()   # paragraphs + headers
  cat('word tokens...\n')
  dtm_w <- task1_nlp_fn(parsed)
  cat('bigram tokens...\n')
  dtm_b <- task2_bigram_fn(parsed)
  save(dtm_w, dtm_b, file = cache)
}

ids <- intersect(dtm_w$.id, dtm_b$.id)
dtm_w <- dtm_w %>% filter(.id %in% ids) %>% mutate(bclass = factor(bclass, levels = lev))
dtm_b <- dtm_b %>% filter(.id %in% ids) %>% mutate(bclass = factor(bclass, levels = lev))
cat('pages:', length(ids), '\n')
cat('word terms:', ncol(dtm_w) - 2, '| bigram terms:', ncol(dtm_b) - 2, '\n')

## 2. HELPERS ---------------------------------------------------------------------
get_xy <- function(dtm, ids){
  d <- dtm[match(ids, dtm$.id), ]
  list(x = as.matrix(select(d, -.id, -bclass)), y = d$bclass)
}

# PCA fit on training rows only; returns train and test scores
pca_scores <- function(x_tr, x_te, m){
  pca <- prcomp(x_tr, center = TRUE, scale. = FALSE, rank. = m)
  list(tr = pca$x[, seq_len(m), drop = FALSE],
       te = predict(pca, x_te)[, seq_len(m), drop = FALSE])
}

# k-fold CV accuracy of a logistic regression on predictors X
cv_accuracy <- function(X, y, fold){
  X <- as.data.frame(X); X$y <- y
  mean(map_dbl(sort(unique(fold)), function(f){
    m <- suppressWarnings(glm(y ~ ., data = X[fold != f, ], family = binomial))
    p <- suppressWarnings(predict(m, X[fold == f, ], type = 'response'))
    mean((p > 0.5) == (X$y[fold == f] == lev[2]))
  }))
}

# out-of-fold predicted log-odds (so stage 2 doesn't see overfit stage-1 output)
oof_logodds <- function(X, y, fold){
  X <- as.data.frame(X); X$y <- y
  out <- numeric(nrow(X))
  for(f in sort(unique(fold))){
    m <- suppressWarnings(glm(y ~ ., data = X[fold != f, ], family = binomial))
    out[fold == f] <- suppressWarnings(predict(m, X[fold == f, ], type = 'link'))
  }
  out
}

test_metrics <- function(truth, prob){
  pred <- factor(ifelse(prob > 0.5, lev[2], lev[1]), levels = lev)
  tibble(accuracy = accuracy_vec(truth, pred),
         sensitivity = sens_vec(truth, pred, event_level = 'second'),
         specificity = spec_vec(truth, pred, event_level = 'second'),
         auc = roc_auc_vec(truth, prob, event_level = 'second'))
}

## 3. TWO-STAGE MODEL ---------------------------------------------------------------
ks <- c(10, 25, 50, 100, 150, 200)  # word PCs to try (stage 1)
ms <- c(2, 5, 10, 25, 50)          # bigram PCs to try (stage 2)

run_split <- function(s){
  set.seed(s)
  labels <- dtm_w %>% select(.id, bclass)
  sp <- initial_split(labels, prop = 0.8, strata = bclass)
  tr_ids <- training(sp)$.id
  te_ids <- testing(sp)$.id
  w_tr <- get_xy(dtm_w, tr_ids); w_te <- get_xy(dtm_w, te_ids)
  b_tr <- get_xy(dtm_b, tr_ids); b_te <- get_xy(dtm_b, te_ids)
  y_tr <- w_tr$y
  fold <- sample(rep(1:5, length.out = length(tr_ids)))

  # stage 1: word-token logistic PCR
  zw <- pca_scores(w_tr$x, w_te$x, max(ks))
  cv1 <- map_dbl(ks, ~ cv_accuracy(zw$tr[, seq_len(.x), drop = FALSE], y_tr, fold))
  k1 <- ks[which.max(cv1)]
  d1 <- as.data.frame(zw$tr[, seq_len(k1), drop = FALSE]); d1$y <- y_tr
  m1 <- suppressWarnings(glm(y ~ ., data = d1, family = binomial))
  te1 <- as.data.frame(zw$te[, seq_len(k1), drop = FALSE])
  lo_te <- suppressWarnings(predict(m1, te1, type = 'link'))
  p1 <- plogis(lo_te)
  lo_tr <- oof_logodds(zw$tr[, seq_len(k1), drop = FALSE], y_tr, fold)

  # stage-1 log-odds can be extreme (out-of-fold values reached several hundred),
  # which destabilizes stage 2; clip to [-10, 10] (probabilities ~ 0.00005 to 0.99995)
  clip <- function(x) pmax(pmin(x, 10), -10)
  lo_tr <- clip(lo_tr)
  lo_te <- clip(lo_te)

  # stage 2: stage-1 log-odds + bigram PCs
  zb <- pca_scores(b_tr$x, b_te$x, max(ms))
  make_X <- function(lo, z, m) data.frame(logodds = lo, z[, seq_len(m), drop = FALSE])
  cv2 <- map_dbl(ms, ~ cv_accuracy(make_X(lo_tr, zb$tr, .x), y_tr, fold))
  m2 <- ms[which.max(cv2)]
  d2 <- make_X(lo_tr, zb$tr, m2); d2$y <- y_tr
  fit2 <- suppressWarnings(glm(y ~ ., data = d2, family = binomial))
  p2 <- suppressWarnings(predict(fit2, make_X(lo_te, zb$te, m2), type = 'response'))

  # diagnostics: check the stage-2 fit is numerically sane
  cat(sprintf(paste0('  [split %d] k1=%d m2=%d | converged=%s | max|coef|=%.1f | ',
                     'logodds coef=%.2f | lo_tr range [%.1f, %.1f] | lo_te range [%.1f, %.1f]\n'),
              s, k1, m2, fit2$converged, max(abs(coef(fit2))), coef(fit2)['logodds'],
              min(lo_tr), max(lo_tr), min(lo_te), max(lo_te)))
  cat('  stage-1 CV acc:', round(cv1, 3), '\n')
  cat('  stage-2 CV acc:', round(cv2, 3), '\n')

  # likelihood-ratio test on training data: do the bigram PCs add anything
  # beyond the stage-1 log-odds?
  base <- suppressWarnings(glm(y ~ logodds, data = d2[, c('logodds', 'y')], family = binomial))
  lr_p <- anova(base, fit2, test = 'Chisq')$`Pr(>Chi)`[2]

  bind_rows(
    test_metrics(w_te$y, p1) %>% mutate(model = 'words only', n_pc = k1, lr_p = NA_real_),
    test_metrics(w_te$y, p2) %>% mutate(model = 'words + bigram PCs', n_pc = m2, lr_p = lr_p)
  ) %>% mutate(seed = s)
}

seeds <- 1:3
results <- map_dfr(seeds, function(s){ cat('split', s, '...\n'); run_split(s) })

## 4. SUMMARY -------------------------------------------------------------------------
cat('\n--- results by split ---\n')
print(results %>% arrange(seed, model), n = Inf)

cat('\n--- average over splits ---\n')
print(results %>%
        group_by(model) %>%
        summarize(across(c(accuracy, sensitivity, specificity, auc), mean),
                  .groups = 'drop'))

cat('\n--- paired difference (stacked minus words only) ---\n')
diffs <- results %>%
  select(seed, model, accuracy, auc) %>%
  pivot_wider(names_from = model, values_from = c(accuracy, auc)) %>%
  mutate(acc_diff = `accuracy_words + bigram PCs` - `accuracy_words only`,
         auc_diff = `auc_words + bigram PCs` - `auc_words only`)
print(diffs %>% select(seed, acc_diff, auc_diff))
cat('mean accuracy difference:', round(mean(diffs$acc_diff), 4),
    '| mean AUC difference:', round(mean(diffs$auc_diff), 4), '\n')

save(results, diffs, file = 'task2-results.RData')
write_csv(results, 'task2-results.csv')
