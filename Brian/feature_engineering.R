rm(list=ls())

PROJ_PATH <- '~/Documents/kaggle/house_prices'
load(file.path(PROJ_PATH, 'data/house_prices.Rdata'))

# Seperate out response variable
sale_price <- train[,'SalePrice']
train['SalePrice'] <- NULL

#--------------------------------------------------------
# Remove variables
#--------------------------------------------------------

# Variables that are over 90% missing
missing_perc <- sapply(train, function(x) sum(is.na(x)) / length(x))
train <- train[,missing_perc <= 0.90]

# Low information
train$Street <- NULL

# One each in pairs of highly correlated variables
train$GarageCars <- NULL
train$GarageYrBlt <- NULL

#--------------------------------------------------------
# Simplify/transform variables
#--------------------------------------------------------

# MSSubClass
mssublcass_map <- list(
  list(c('20','30','40','120'), 1),
  list(c('45','50','150'), 1.5),
  list(c('60','70','160'), 2),
  list(c('75'), 2.5)
)

train$Floors <- as.numeric(NA)
for (mp in mssublcass_map){
  train$Floors[train$MSSubClass %in% mp[[1]]] <- mp[[2]]
}

train$shared <- with(train, ifelse(MSSubClass %in% c('90','190'), 1, 0))
train$MSSubClass <- NULL

# MSZoning
train$Residential <- ifelse(substr(train$MSZoning,1,1)=='R', 1, 0)

mszoning_map <- list(
  list(c('A','RL','RP'), 0),
  list(c('RM','FV', 'I'), 1),
  list(c('C','C (all)','RH'), 2)
)

train$Density <- NA
for (mp in mszoning_map){
  train$Density[train$MSZoning %in% mp[[1]]] <- mp[[2]]
}
train$MSZoning <- NULL

# Lot Area
train$LogLotArea <- log(train$LotArea)
train$LotArea <- NULL

# Lot Shape
train$LotShapeIrregular <- ifelse(train$LotShape=='Reg', 0,
                                  as.numeric(substr(train$LotShape,3,3)))
train$LotShape <- NULL


# Count a finished basement in square footage
train$GrLivArea <- with(train, GrLivArea + BsmtFinSF1 + BsmtFinSF2)
train$BsmtFinSF1 <- NULL
train$BsmtFinSF2 <- NULL



nm <- names(train)
cl <- sapply(train, 'class')

#--------------------------------------------------------
# Fill in missing data
#--------------------------------------------------------

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
