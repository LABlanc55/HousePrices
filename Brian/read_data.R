rm(list=ls())

PROJ_PATH <- '~/Documents/kaggle/house_prices'

train <- read.csv(file.path(PROJ_PATH, 'data/train.csv'))
test <- read.csv(file.path(PROJ_PATH, 'data/test.csv'))

nm <- names(train)

# Numeric variable indices
num_idx <- c(4,5,18,19,20,21,27,35,37,38,39,44,45,46,47,48,49,50,51,
             52,53,55,57,60,62,63,67,68,69,70,71,72,76,77,78)

# Categorical variable indices
cat_idx <- setdiff(2:(ncol(train)-1), num_idx)

# Make categorical variables factors
for (idx in cat_idx){
  train[[idx]] <- factor(train[[idx]])
  test[[idx]] <- factor(test[[idx]])
}

# Save to R dataset
save(list=ls(), file=file.path(PROJ_PATH, 'data/house_prices.Rdata'))
