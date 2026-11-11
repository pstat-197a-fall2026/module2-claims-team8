suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(textstem)
  library(rvest)
  library(qdapRegex)
  library(stopwords)
  library(tokenizers)
})

# parse html and clean text.

task1_parse_fn <- function(.html, selector = 'p, h1, h2, h3, h4, h5, h6'){
  read_html(.html) %>%
    html_elements(css = selector) %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# apply to claims data
task1_parse_data <- function(.df, selector = 'p, h1, h2, h3, h4, h5, h6'){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = task1_parse_fn(text_tmp, selector)) %>%
    unnest(text_clean) %>%
    ungroup()
  return(out)
}

# tokenize, lemmatize, tf-idf (wide document-term matrix).
# Terms appearing in fewer than min_df documents are dropped to keep the
# matrix manageable.
task1_nlp_fn <- function(parse_data.out, min_df = 5){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    group_by(token.lem) %>%
    filter(n_distinct(.id) >= min_df) %>%
    ungroup() %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}
