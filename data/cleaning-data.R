## builds shared cleaned data + train/validation split
## run once (or again whenever preprocessing.R changes)
library(tidyverse)
library(tidymodels)
source('scripts/preprocessing-primary.R')
load('data/claims-raw.RData')

claims_clean <- claims_raw %>% parse_data()
save(claims_clean, file = 'data/claims-clean.RData')

# quick checks
nrow(claims_clean)                  # should be 2165
sum(claims_clean$text_clean == '')  # pages with no usable text

# shared split -- everyone uses this
set.seed(2026)
partitions <- initial_split(claims_clean, prop = 0.8, strata = mclass)
save(partitions, file = 'data/claims-partitions.RData')