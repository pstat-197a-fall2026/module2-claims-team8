## TASK 1: does adding header text improve binary predictions

install.packages(c("tidyverse","tidymodels","tidytext","textstem","rvest","qdapRegex","stopwords","tokenizers"))

source('task1_preprocessing.R')
suppressPackageStartupMessages(library(tidymodels))

# find a data file whether or not it sits in a data/ subfolder
find_file <- function(f){
  for(d in c('data', '.')){
    p <- file.path(d, f)
    if(file.exists(p)) return(p)
  }
  stop('cannot find ', f)
}

load(find_file('claims-raw.RData'))   # creates claims_raw
lev <- levels(factor(claims_raw$bclass))
cat('binary class levels:', lev, '\n')

## 1. PREPROCESS: paragraph-only vs paragraph + headers ----------------------
cache <- 'task1-cache.RData'
if(file.exists(cache)){
  load(cache)
}else{
  cat('parsing paragraph-only...\n')
  dtm_p <- claims_raw %>%
    task1_parse_data(selector = 'p') %>%
    task1_nlp_fn()
  cat('parsing paragraph + headers...\n')
  dtm_ph <- claims_raw %>%
    task1_parse_data(selector = 'p, h1, h2, h3, h4, h5, h6') %>%
    task1_nlp_fn()
  save(dtm_p, dtm_ph, file = cache)
}

# use the same pages for both versions so the comparison is fair
ids <- intersect(dtm_p$.id, dtm_ph$.id)
dtm_p  <- dtm_p  %>% filter(.id %in% ids) %>% mutate(bclass = factor(bclass, levels = lev))
dtm_ph <- dtm_ph %>% filter(.id %in% ids) %>% mutate(bclass = factor(bclass, levels = lev))
cat('pages:', length(ids), '\n')
cat('terms (paragraph only):', ncol(dtm_p) - 2, '\n')
cat('terms (paragraph + headers):', ncol(dtm_ph) - 2, '\n')

## 2. LOGISTIC PCR ------------------------------------------------------------
ks <- c(10, 25, 50, 100, 150, 200)   # candidate numbers of PCs

fit_pcr <- function(dtm, train_ids, test_ids, ks, nfold = 5){
  get_xy <- function(ids){
    d <- dtm[match(ids, dtm$.id), ]
    list(x = as.matrix(select(d, -.id, -bclass)), y = d$bclass)
  }
  tr <- get_xy(train_ids)
  te <- get_xy(test_ids)

  # PCA on training data only; project test data onto the same PCs
  pca <- prcomp(tr$x, center = TRUE, scale. = FALSE, rank. = max(ks))
  z_tr <- pca$x[, seq_len(max(ks)), drop = FALSE]
  z_te <- predict(pca, te$x)[, seq_len(max(ks)), drop = FALSE]

  # choose number of PCs by CV accuracy within the training set
  fold <- sample(rep(seq_len(nfold), length.out = nrow(z_tr)))
  cv_acc <- map_dbl(ks, function(k){
    mean(map_dbl(seq_len(nfold), function(f){
      d <- as.data.frame(z_tr[, seq_len(k), drop = FALSE])
      d$y <- tr$y
      m <- suppressWarnings(glm(y ~ ., data = d[fold != f, ], family = binomial))
      p <- suppressWarnings(predict(m, d[fold == f, ], type = 'response'))
      mean((p > 0.5) == (d$y[fold == f] == lev[2]))
    }))
  })
  k_best <- ks[which.max(cv_acc)]

  # refit with chosen k and evaluate on held-out test pages
  d <- as.data.frame(z_tr[, seq_len(k_best), drop = FALSE]); d$y <- tr$y
  m <- suppressWarnings(glm(y ~ ., data = d, family = binomial))
  dte <- as.data.frame(z_te[, seq_len(k_best), drop = FALSE])
  p <- suppressWarnings(predict(m, dte, type = 'response'))

  res <- tibble(truth = te$y,
                prob = p,
                pred = factor(ifelse(p > 0.5, lev[2], lev[1]), levels = lev))
  tibble(k = k_best,
         accuracy = accuracy_vec(res$truth, res$pred),
         sensitivity = sens_vec(res$truth, res$pred, event_level = 'second'),
         specificity = spec_vec(res$truth, res$pred, event_level = 'second'),
         auc = roc_auc_vec(res$truth, res$prob, event_level = 'second'))
}

## 3. REPEAT OVER SEVERAL RANDOM 80/20 SPLITS ----------------------------------
seeds <- 1:3
labels <- dtm_p %>% select(.id, bclass)

results <- map_dfr(seeds, function(s){
  set.seed(s)
  sp <- initial_split(labels, prop = 0.8, strata = bclass)
  tr_ids <- training(sp)$.id
  te_ids <- testing(sp)$.id
  cat('split', s, '...\n')
  bind_rows(
    fit_pcr(dtm_p,  tr_ids, te_ids, ks) %>% mutate(version = 'paragraphs only'),
    fit_pcr(dtm_ph, tr_ids, te_ids, ks) %>% mutate(version = 'paragraphs + headers')
  ) %>% mutate(seed = s)
})

## 4. SUMMARY -------------------------------------------------------------------
cat('\n--- results by split ---\n')
print(results %>% arrange(seed, version), n = Inf)

cat('\n--- average over splits ---\n')
summ <- results %>%
  group_by(version) %>%
  summarize(across(c(accuracy, sensitivity, specificity, auc), mean),
            mean_k = mean(k), .groups = 'drop')
print(summ)

cat('\n--- paired difference in accuracy (headers minus paragraphs only) ---\n')
diffs <- results %>%
  select(seed, version, accuracy) %>%
  pivot_wider(names_from = version, values_from = accuracy) %>%
  mutate(diff = `paragraphs + headers` - `paragraphs only`)
print(diffs)
cat('mean difference:', round(mean(diffs$diff), 4), '\n')

save(results, summ, diffs, file = 'task1-results.RData')
write_csv(results, 'task1-results.csv')
