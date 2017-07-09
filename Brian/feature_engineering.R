rm(list=ls())

PROJ_PATH <- '~/Documents/kaggle/house_prices'
load(file.path(PROJ_PATH, 'data/house_prices.Rdata'))

# Seperate out response variable
sale_price <- train[,'SalePrice']
train['SalePrice'] <- NULL

#--------------------------------------------------------
# Remove variables
#--------------------------------------------------------

train$Id <- NULL

# Variables that are over 90% missing
missing_perc <- sapply(train, function(x) sum(is.na(x)) / length(x))
train <- train[,missing_perc <= 0.90]

# Low or redundant information
train$Street <- NULL
train$Utilities <- NULL
train$HouseStyle <- NULL
train$RoofMatl <- NULL
train$EnclosedPorch <- NULL
train$X3SsnPorch <- NULL
train$ScreenPorch <- NULL
train$PoolArea <- NULL
train$MiscVal <- NULL
train$PoolQC <- NULL

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

# LotArea
train$LogLotArea <- log(train$LotArea)
train$LotArea <- NULL

# LotShape
train$LotShapeIrregular <- ifelse(train$LotShape=='Reg', 0,
                                  suppressWarnings(as.numeric(substr(train$LotShape,3,3))))

train$LotShape <- NULL

# LandContour
train$Level <- ifelse(train$LandContour=='Lvl', 1, 0)
train$LandContour <- NULL

# LotConfig
train$InsideLot <- ifelse(train$LotConfig=='Inside', 1, 0)
train$LotConfig <- NULL

# Neighborhood
# Could try to do some clustering to reduce categories

# Noise
noise_map <- list(
  high=c('Artery','RRAn','RRAe'),
  medium=c('Feedr','RRNn','RRNe'),
  low=c('Norm','PosN')
)

train$NoiseLevel <- factor(NA, levels=names(noise_map))

for (lv in names(noise_map)){
  bool <- train$Condition1 %in% noise_map[[lv]] |
           train$Condition2 %in% noise_map[[lv]]
  train$NoiseLevel[bool & is.na(train$NoiseLevel)] <- lv
}

train$Condition1 <-NULL
train$Condition2 <- NULL

# BldgType
train$BldgType[train$BldgType=='2FmCon'] <- 'Duplex'

#YearBuilt & YearRemodAdd
train$LogBldgAge <- log(2017 - train$YearBuilt)
train$LogBldgYearsToRemod <- ifelse(train$YearRemodAdd == train$YearBuilt, 0,
                                 log(train$YearRemodAdd - train$YearBuilt))
train$YearBuilt <- NULL
train$YearRemodAdd <- NULL

# RoofStyle
levels(train$RoofStyle) <- c(levels(train$RoofStyle), 'Other')
train$RoofStyle[
  train$RoofStyle %in% c('Shed','Mansard','Gambrel','Flat')] <- 'Other'

# Exterior2nd
train$Exterior2nd[
  as.character(train$Exterior2nd) == as.character(train$Exterior1st)] <- NA

# BsmtQual
train$BsmtQual <- factor(train$BsmtQual, levels=c('Po','Fa','TA','Gd','Ex'))
train$BsmtQual <- as.numeric(train$BsmtQual)


# Revisit basement square footage

# Count a finished basement in square footage
train$GrLivArea <- with(train, GrLivArea + BsmtFinSF1 + BsmtFinSF2)
train$BsmtFinSF1 <- NULL
train$BsmtFinSF2 <- NULL

# Heating
train$GasHeat <- ifelse(grepl('Gas', train$Heating), 1, 0)
train$Heating <- NULL

# Electricity
train$Elctr <- as.character(train$Electrical)
train$Elctr[train$Electrical %in% c('FuseA','Mix')] <- 'FuseGood'
train$Elctr[train$Electrical %in% c('FuseF','FuseP')] <- 'FuseGood'
train$Elctr[is.na(train$Electrical)] <- 'SBrkr'

train$Elctr <- factor(train$Elctr)
train$Electrical <- NULL

# Above ground square feet
train$Log1stFlrSF <- log(train$X1stFlrSF)
train$X1stFlrSF <- NULL

# Functional
train$Functional <- as.integer(train$Functional == 'Typ')

# GarageType
train$GarageType[train$GarageType %in% c('2Types','Basment','BuiltIn')] <- 'Attchd'
train$GarageType[train$GarageType == 'CarPort'] <- NA

# SaleType
train$SaleType[!(train$SaleType %in% c('WD','New'))] <- 'Oth'

# SaleCondtion
train$SaleCondition[train$SaleCondition %in% c('AdjLand','Family','Alloca')] <- 'Abnorml'

#--------------------------------------------------------
# Fill in missing data
#--------------------------------------------------------

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
