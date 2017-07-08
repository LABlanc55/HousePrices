library(mice)
library(glmnet)
library(randomForest)

rm(list=ls())

PROJ_PATH <- '~/Documents/kaggle/house_prices'

K <- 2
MAX_TEST_PERC <- 0.20
SEED <- 1e5

rmse <- function(x, y){
  if(x && length(x)==length(y))
    sqrt(mean((x-y)^2))
  else
    stop('ERROR: rmse')
}

plot_pred <- function(..., col=adjustcolor('gray30', alpha.f=0.4), bty='n'){
  lims <- range(unlist(lapply(list(...)[1:2], range)))
  plot(..., col=col, bty=bty, xlim=c(lims[1],lims[2]), ylim=c(lims[1],lims[2]))
  abline(a=0, b=1, col='orange')
}

load(file.path(PROJ_PATH, 'data/model_data.Rdata'))

#--------------------------------------------------------
# Sets for cross validation
#--------------------------------------------------------

set.seed(SEED)

nr <- nrow(train)
cv <- sample(1:nr, nr * min(MAX_TEST_PERC*K, 1), replace=FALSE)
cv <- matrix(cv, K)

#--------------------------------------------------------
# Random Forest
#--------------------------------------------------------

rf <- list()
set.seed(SEED)

for (i in 1:K){ #need to do imputation separately on test data
  test_idx <- cv[i,]
  train_idx <- setdiff(1:nr, test_idx)
  
  rf[[i]] <- randomForest(x=train[train_idx,],
                          y=sale_price[train_idx],
                          xtest=train[test_idx,],
                          ytest=sale_price[test_idx])
  
  rf[[i]]$rmse <- rmse(log(rf[[i]]$test$predicted),
                       log(sale_price[test_idx]))
  message(rf[[i]]$rmse)
  
  plot_pred(log(sale_price[test_idx]),
            log(rf[[i]]$test$predicted),
            main=paste('Fold', i),
            xlab='Actual',
            ylab='Predicted')
}

#--------------------------------------------------------
# LASSO
#--------------------------------------------------------
