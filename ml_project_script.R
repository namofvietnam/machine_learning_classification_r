# Load the data 
df = read.csv("~/cs-training.csv")

#################
## Manage Data ##
#################

df$SeriousDlqin2yrs = as.logical(df$SeriousDlqin2yrs)
df$X = NULL
# Filling missing data

## Nassim Nicolas Taleb identified two areas for variables, mediocristan and extremistan
## In mediocristan, values of a variable do not deviate significantly from the mean, and an additional observation would not change the standard deviation significantly. Examples include biometrics and physics-bound variables
## Values of a variable in extremistan deviate widely from the mean, and an additional variable can significantly alter the standard deviation. These variables often have scalability, for example revenue, income, virus spread.
## In our data set, age and number of dependents likely fall into mediocristan. People have life expectancy, and in general do not differ widely on the number of dependents.
## Thankfully, there was no missing value in age, but there was an instance of age 0, which we treated as a missing age value.
df$age[df$age == 0] <- NA
## Because mediocristan variables typically follow distributions found in nature, we fill the missing number of dependents values based on its existing distribution in the original data set.
## Being a liquid and convenient representation of value, money is highly scalable, and thus money-related variables like monthly income likely fall into extremistan.
## As extremistan variables do not conform to normal distributions, we could not fill missing monthly income values with a distribution.
## Predictive models are typically better at detecting possible relationships and associations in situations where a distribution of a target variable is unstable.
## Therefore, we used a predictive model called "Random Forest" to predict missing monthly income values, based on other variables (more on Random Forest later).
## Looking at the dataset, we noticed that debt ratio, the ratio of credit card debt amount over monthly income, had abnormal values (e.g., 477, 5710, instead of .4, .6, etc.) wherever monthly income is missing, because this ratio depends on the monthly income.
## These abnormal debt ratios are most likely credit card balance, unconverted to debt ratios due to missing monthly income values
## Therefore, we keep the abnormal debt ratio values, because they would be used to estimate debt ratio after missing monthly income values are imputed.
## In the prediction of monthly income, therefore, debt ratio cannot play a part. Abnormal debt ratio values would be replaced with themselves (presumably credit card balance) divided by imputed montly income.
## The interdependence between monthly income and debt ratio render it implausible to distribute number of dependents based on either of these values. 
## Nor should we base number of dependents on monthly income and debt ratio. 
## We also want to separate the prediction of missing number of dependents from financial variables, with the assumption that dependency situations can occur due to other factors than finance, but are still associated with age.
## Therefore, use bootstrapping sampling to fill number of dependents in a separate dataframe  consisting of only age and number of dependents.
## This way, number of dependents are distributed according to age.

library(mice)

set.seed(1234)
impute_NoD = mice(data.frame(df$age, df$NumberOfDependents),method="sample")
df_imputed_NoD = complete(impute_NoD)
df$NumberOfDependents = df_imputed_NoD$df.NumberOfDependents
df$age = df_imputed_NoD$df.age

## Only then could we include number of dependents in the predictive model to fill missing values for monthly income.
## We fill missing values in monthly income along with those in other variables, except for Debt Ratio
## Finally, with the filled monthly income and the preserved amount of credit card debt, we could calculate the correct debt ratio.

# We assume that 0 monthly income represents those without income that were somehow approved for a credit card. To avoid indefinite values in the debt ratio calculation later, we substitute 0 with 1, to represent no income. Anything lower than 1 would unneccesarily increase the debt ratio.
df$MonthlyIncome[df$MonthlyIncome == 0] <- 1 
idx_na_income = is.na(df$MonthlyIncome)
set.seed(1234)
impute_rf = mice(df[ , -which(colnames(df) == "DebtRatio")],method='rf') # random forest
df_imputed_income = complete(impute_rf)
df$MonthlyIncome = df_imputed_income$MonthlyIncome
df[idx_na_income,"DebtRatio"] = df[idx_na_income,"DebtRatio"]/df[idx_na_income,"MonthlyIncome"]

backup_clean_df = df
#write.csv(backup_clean_df,"/Volumes/GoogleDrive/My Drive/Studies/Machine Learning/Project/backup_clean_df.csv") #save the clean data!

##########################
### Partition the data ###
##########################
set.seed(1234) # We set this seed to make the modelling process repeatable. This seed did not result in any anomaly within and between the training and test data sets.

trainingCases = sample(nrow(df), round(nrow(df)*0.6))
training = df[trainingCases,]
test = df[-trainingCases,]

obs = test$SeriousDlqin2yrs # the true observations in the test dataset
error_bench = benchmarkErrorRate(training$SeriousDlqin2yrs,test$SeriousDlqin2yrs)
beta = 0.5
highest_allowable_cutoff = .0835 # adjustable, depending on the decision maker's theory

#################
### Functions ###
#################

# The following functions were written by Nathan Karst, professor at Babson College, and not my own codes.
# They are required for this project.
## benchmark error rate
benchmarkErrorRate <- function(training, test){
  prop_train = as.data.frame(prop.table(table(training)))
  prop_train=prop_train[order(-prop_train$Freq),]
  dominant_class=prop_train[1,1]
  guess = as.character(dominant_class)
  percent_correct_simple=sum(guess == as.character(test))/length(test)
  return(1 - percent_correct_simple)
}

## easyPrune
easyPrune <- function(model){ 
  return(prune(model, cp = model$cptable[which.min(model$cptable[ , "xerror"]), "CP"]))
} ### with cross val?

## Cross Validation to choose best k
kNN = function (form, train, test, standardize = T, ...) 
{
  out <- tryCatch(
    {
      library(class)
    },
    error=function(cond) {
      install.packages("class")
      library(class)
    })    
  
  tgtCol <- which(colnames(train) == as.character(form[[2]]))
  if (standardize) {
    tmp <- scale(train[, -tgtCol], center = T, scale = T)
    train[, -tgtCol] <- tmp
    ms <- attr(tmp, "scaled:center")
    ss <- attr(tmp, "scaled:scale")
    test[, -tgtCol] <- scale(test[, -tgtCol], center = ms, 
                             scale = ss)
  }
  return(knn(train[, -tgtCol], test[, -tgtCol], unlist(train[, tgtCol]), ...))
}

## ROCCchart
ROCChart <- function(obs, pred){ 
  out <- tryCatch(
    {
      library(ggplot2)
    },
    error=function(cond) {
      install.packages("ggplot2")
      library(ggplot2)
    })  
  
  se = c()
  sp = c()
  P = c()
  for(i in 1:101){
    p = (i-1)/100
    
    predTF = (pred > p)
    
    sp[i] = sum(predTF == FALSE & obs == FALSE)/sum(obs == FALSE)
    se[i] = sum(predTF == TRUE & obs == TRUE)/sum(obs == TRUE)
    P[i] = p
  } 
  
  df1 = data.frame(x=1-sp,y=se,"type"="Observed")
  df1 = df1[order(df1$x,df1$y),]
  
  df2 = data.frame(x=1-sp,y=1-sp,"type"="Benchmark")
  df3 = data.frame(x=c(0,0,1),y=c(0,1,1),"type"="Ideal")
  
  
  df = rbind(df2,df3,df1)
  
  qplot(x,y,data=df,color=type,geom=c("point","line"),ylab="Sensitivity = True Positive Rate",xlab="1 - Specificity = False Positive Rate",main="ROC Chart")
  
}

easyPrune <- function(model){ 
  return(prune(model, cp = model$cptable[which.min(model$cptable[ , "xerror"]), "CP"]))
}

###########################
### Random Forest Model ###
###########################

library(randomForest)
set.seed(1234)
rf = randomForest(SeriousDlqin2yrs ~., data = training, ntree = 500)
rf_pred = predict(rf, test)

# Drawing insights: Variable importance
# In a random forest model, “Gini impurity” or “node impurity” measures how frequently a randomly chosen feature would be mislabeled if it were randomly identified based on the random distribution of observations in that variable. 
# Impurity increases with randomness. 
# A variable that increases in purity during the random selection process of the forest shall prove to be more important. 

varImpPlot(rf) 

# Evaluation of predictions:
rf_pred_TF = rf_pred > highest_allowable_cutoff
rf_sensitivity = sum(rf_pred_TF == TRUE & obs == TRUE)/sum(obs == TRUE)
rf_precision = sum(rf_pred_TF == TRUE & obs == TRUE)/sum(rf_pred_TF == TRUE)
rf_F_beta = ((1+beta^2)*(rf_sensitivity*rf_precision))/((beta^2)*rf_precision + rf_sensitivity)

# Find the optimal F-value and cut-off probability
find_optimal_cutoff_for_highest_F <- function(predictions, beta){
  F_beta_set = data.frame(matrix(ncol = 2, nrow = 0))
  for (cutoff_p in seq(0,1, by = 0.0001)) {
    pred_TF = predictions > cutoff_p
    sensitivity = sum(pred_TF == TRUE & obs == TRUE)/sum(obs == TRUE)
    precision = sum(pred_TF == TRUE & obs == TRUE)/sum(pred_TF == TRUE)
    
    F_beta = (1+beta^2)*(sensitivity*precision)/((beta^2)*precision+sensitivity)
    F_beta_set = rbind(F_beta_set,data.frame(cutoff_p,F_beta))
  }
  no_nan_F_set = F_beta_set[!is.nan(F_beta_set$F_beta),]
  max_F = max(no_nan_F_set$F_beta)
  optimal_cutoff_p = no_nan_F_set[no_nan_F_set$F_beta == max_F,"cutoff_p"]
  plot(F_beta_set)
  return(list(max_F = max_F, optimal_cutoff_p = optimal_cutoff_p))
}

rf_optimal_combination = find_optimal_cutoff_for_highest_F(predictions = rf_pred,beta = beta)
rf_max_F = rf_optimal_combination$max_F
rf_optimal_cutoff_p = rf_optimal_combination$optimal_cutoff_p
  
##################################
### Logistics Regression Model ###
##################################

model_Logistics = glm(SeriousDlqin2yrs ~., data = training, family = 'binomial') #glm general linear model, choose family binomial for logistic regression)
model_Logistics = step(model_Logistics)
summary(model_Logistics)

## Evaluation of the Logistics Regression Model

log_pred = predict(model_Logistics, test, type = 'response')

log_optimal_combination = find_optimal_cutoff_for_highest_F(log_pred, beta)
log_max_F = log_optimal_combination$max_F
log_optimal_cutoff_p = log_optimal_cutoff_p$optimal_cutoff_p

# found cutoff probability that is higher than allowed.
# Therefore, we need to lower the cut-off to the highest allowable.
log_pred_TF = log_pred > highest_allowable_cutoff
table(log_pred_TF, obs)
log_sensitivity = sum(log_pred_TF == TRUE & obs == TRUE)/sum(obs == TRUE)
log_precision = sum(log_pred_TF == TRUE & obs == TRUE)/sum(log_pred_TF == TRUE)
log_F_beta = (1+beta^2)*(log_sensitivity*log_precision)/((beta^2)*log_precision+log_sensitivity)

################
### Boosting ###
################

library(gbm)
set.seed(1234)
boost = gbm(SeriousDlqin2yrs ~ ., data=training, n.trees = 1000, cv.folds = 4)
set.seed(1234)
best_size = gbm.perf(boost) #plot model complexity chart

boost_pred = predict(boost, test, best_size, type = "response")
boost_optimal_combination = find_optimal_cutoff_for_highest_F(boost_pred, beta)
boost_max_F = boost_optimal_combination$max_F
boost_optimal_cutoff_p = boost_optimal_combination$optimal_cutoff_p

boost_pred_TF = boost_pred > highest_allowable_cutoff

boost_sensitivity = sum(boost_pred_TF == TRUE & obs == TRUE)/sum(obs == TRUE)
boost_precision = sum(boost_pred_TF == TRUE & obs == TRUE)/sum(boost_pred_TF == TRUE)

boost_F_beta = (1+beta^2)*(boost_sensitivity*boost_precision)/((beta^2)*boost_precision+boost_sensitivity)

# Variable importance as measured by permutation of randomness 
# (introducing randomness to figure out variables' usefulness in the model
# same idea with introducing randomness to observe "purity" in rf)
par(mar = c(5, 18, 1, 1))
summary(
  boost_best_n_trees, 
  cBars = 10,
  method = permutation.test.gbm, # can use permutation.test.gbm (permutation), relative.influence (please look up this method if you would like)
  las = 2
)

################
### STACKING ###
################
# Create a battery of models, in which one manager model evaluate the other models

# Create a stacked data frame with predictions by 3 models
pred_rf_full = predict(rf, df)
pred_log_full = predict(model_Logistics,df,type = 'response')
# pred_kNN_full = predict(model_kNN, df)
pred_boost_full = predict(boost, df, best_size, type = "response")

df_stacked = cbind(df, pred_rf_full, pred_log_full, pred_boost_full)
train_stacked = df_stacked[trainingCases, ]
test_stacked = df_stacked[-trainingCases, ]

set.seed(1234)
subtest_cases = sample(nrow(test_stacked), round(nrow(test_stacked)*0.5))
subtest_stacked = test_stacked[subtest_cases, ] 
sub_obs = subtest_stacked$SeriousDlqin2yrs

#1. Stacking with logistics regression
stack_log = glm(SeriousDlqin2yrs ~ .,data = train_stacked,family = binomial) %>% step()
stack_log_pred = predict(stack_log, subtest_stacked, type = 'response')

stack_log_optimal_combination = find_optimal_cutoff_for_highest_F(stack_log_pred, beta)
stack_log_max_F = stack_log_optimal_combination$max_F
stack_log_optimal_cutoff_p = stack_log_optimal_combination$optimal_cutoff_p

stack_log_pred_TF = stack_log_pred > highest_allowable_cutoff
stack_log_sensitivity = sum(stack_log_pred_TF == TRUE & sub_obs == TRUE)/sum(sub_obs == TRUE)
stack_log_precision = sum(stack_log_pred_TF == TRUE & sub_obs == TRUE)/sum(stack_log_pred_TF == TRUE)
stack_log_F_beta = ((1+beta^2)*(stack_log_sensitivity*stack_log_precision))/((beta^2)*stack_log_precision + stack_log_sensitivity)

#2. Stacking with random forest
set.seed(1234)
stack_rf = randomForest(SeriousDlqin2yrs ~ .,data = train_stacked, ntree = 500)
stack_rf_pred = predict(stack_rf, subtest_stacked)

stack_rf_optimal_combination = find_optimal_cutoff_for_highest_F(stack_rf_pred, beta)
stack_rf_max_F = stack_rf_optimal_combination$max_F
stack_rf_optimal_cutoff_p = stack_rf_optimal_combination$optimal_cutoff_p

stack_rf_pred_TF = stack_rf_pred > highest_allowable_cutoff
stack_rf_sensitivity = sum(stack_rf_pred_TF == TRUE & sub_obs == TRUE)/sum(sub_obs == TRUE)
stack_rf_precision = sum(stack_rf_pred_TF == TRUE & sub_obs == TRUE)/sum(stack_rf_pred_TF == TRUE)
stack_rf_F_beta = (1+beta^2)*(stack_rf_sensitivity*stack_rf_precision)/((beta^2)*stack_rf_precision+stack_rf_sensitivity)

#3. Stacking with Neural Network - expect high performance and a black box
library(caret)
standardize_stacked_df = preProcess(df_stacked, method = c('center','scale'))
standard_train_stacked = predict(standardize_stacked_df,train_stacked)
standard_test_stacked = predict(standardize_stacked_df,test_stacked)
standard_subtest_stacked = predict(standardize_stacked_df,subtest_stacked)

library(nnet)
library(NeuralNetTools)
set.seed(1234)
stack_NN = nnet(SeriousDlqin2yrs ~ ., data = standard_train_stacked, size = 1)

par(mar = numeric(4))
plotnet(stack_NN,pad_x = .5)

stack_NN_pred = predict(stack_NN, subtest_stacked)
sum(stack_NN_pred == 1) # = 0; no positive prediction

stack_NN_optimal_combination = find_optimal_cutoff_for_highest_F(stack_NN_pred, beta)
stack_NN_max_F = stack_NN_optimal_combination$max_F
stack_NN_optimal_cutoff_p = stack_NN_optimal_combination$optimal_cutoff_p


## Applying stacked logistic model to stacked test

stack_finlog_pred = predict(stack_log, test_stacked, type = 'response')

stack_finlog_optimal_combination = find_optimal_cutoff_for_highest_F(stack_finlog_pred, beta)
stack_finlog_max_F = stack_finlog_optimal_combination$max_F
stack_finlog_optimal_cutoff_p = stack_finlog_optimal_combination$optimal_cutoff_p

summary(stack_log)

## Applying Random Forest to Stacked Test
stack_finrf_pred = predict(stack_rf, test_stacked, type = 'response')

stack_finrf_optimal_combination = find_optimal_cutoff_for_highest_F(stack_finrf_pred, beta)
stack_finrf_max_F = stack_finrf_optimal_combination$max_F
stack_finrf_optimal_cutoff_p = stack_finrf_optimal_combination$optimal_cutoff_p

stack_finrf_pred_TF = stack_finrf_pred > highest_allowable_cutoff
stack_finrf_sensitivity = sum(stack_finrf_pred_TF == TRUE & obs == TRUE)/sum(obs == TRUE)
stack_finrf_precision = sum(stack_finrf_pred_TF == TRUE & obs == TRUE)/sum(stack_finrf_pred_TF == TRUE)
stack_finrf_F_beta = (1+beta^2)*(stack_finrf_sensitivity*stack_finrf_precision)/((beta^2)*stack_finrf_precision+stack_finrf_sensitivity)

varImpPlot(stack_rf)
summary(stack_rf)

