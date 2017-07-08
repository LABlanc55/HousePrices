rm(list=ls())


PROJ_PATH <- '~/Documents/kaggle/house_prices'
load(file.path(PROJ_PATH, 'data/house_prices.Rdata'))

# Seperate out response variable
sale_price <- train[,'SalePrice']
train['SalePrice'] <- NULL

# Remove variables that are over 90% missing
missing_perc <- sapply(train, function(x) sum(is.na(x)) / length(x))
train <- train[,missing_perc <= 0.90]

nm <- names(train)
cl <- sapply(train, 'class')

# Replace NA with 'Missing' in remaining categorical variables 
for (i in 1:ncol(train)){
  if (cl[i]=='factor' && any(is.na(train[,i]))){
    levels(train[,i]) <- c(levels(train[,i]), 'Missing')
    train[,i][is.na(train[,i])] <- 'Missing'
  }
}

# Impute numeric variables
set.seed(1e5)
train_num_imp <- complete(mice(train[,cl!='factor'], m=1))

train <- cbind(train[,cl=='factor'], train_num_imp)
train <- train[,nm]

# Export new training set
save(list=c('train','test','sale_price'), file=file.path(PROJ_PATH, 'data/model_data.Rdata'))
