## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text
# now captures headers (h1-h6) AND paragraphs, in page order
parse_fn <- function(.html){
  raw_text <- tryCatch(
    read_html(.html) %>%
      html_elements('h1, h2, h3, h4, h5, h6, p') %>%
      html_text2() %>%
      str_c(collapse = ' '),
    error = function(e) ''   # unreadable page -> empty text instead of crashing
  )
  
  raw_text %>%
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
    str_replace_all("\\s+", " ") %>%
    str_trim()
}

# function to apply to claims data
# no longer drops any rows -- every page gets a text_clean value
parse_data <- function(.df){
  .df %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    ungroup() %>%
    mutate(text_clean = replace_na(text_clean, ''))
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}
