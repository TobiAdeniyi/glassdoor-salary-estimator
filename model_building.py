#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:26:47 2020

@author: tobiadeniyi
"""

#############################################################################################
### Import dependancies ###
#############################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

import ast
import pickle

from xgboost import XGBRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score



#############################################################################################
### Load data frame ###
#############################################################################################

df = pd.read_csv("glassdoor_jobs_eda.csv", index_col=0)



#############################################################################################
### Choose relevent columns ###
#############################################################################################

# Print all columns
i = 0
column_lst = list(df.columns)

for col in column_lst:
    print(i, col)
    i += 1
    
print("\n\n\n")

# Columns we believe are valuable
col_idx = [24, 6, 7, 19, 20, 21, 18, 8, 9, 14, 17, 25, 26, 27, 28, 29]

"""
We dropped seniority  as the junior roles (stated explicitly)
were so sparce, reulting in junior roles having heigh salary.
"""

cols = []
# Select columns
for i in col_idx:
    col = column_lst[i]
    cols.append(col)

df_ml = df[cols]



#############################################################################################
### Preprocessing -- Get dummy data for cat variables ###
#############################################################################################

# Turn elements of list columns to list from strings
df_ml["languages"] = df_ml["languages"].apply(lambda x: ast.literal_eval(x))
df_ml["industry"] = df_ml["industry"].apply(lambda x: ast.literal_eval(x))
df_ml["sector"] = df_ml["sector"].apply(lambda x: ast.literal_eval(x))


# Use multilable encoding to unpack list variables
lst_vars = ["industry", "sector", "languages"]

# Tranform encode each column
for var in lst_vars:
    mlb = MultiLabelBinarizer()
    df_ml = df_ml.join(pd.DataFrame(mlb.fit_transform(df_ml.pop(var)),
            columns=mlb.classes_,index=df_ml.index), lsuffix='_industry', rsuffix='_sector')
    
# Get dummies for transformed dataframe
    
## dummy variables... Help us normalise
df_dm = pd.get_dummies(df_ml)



#############################################################################################
### Create train test split ###
#############################################################################################

# split data into training and validation data, for both features and target
y = df_dm.avg_salary.values
X = df_dm.drop("avg_salary", axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state = 0)




#############################################################################################
### Model building ###
#############################################################################################

############################
# Multiple linear regresion
############################

## Using Stats Models' OLS
X_ols = sm.add_constant(X)
model_ols = sm.OLS(y,X_ols)
results_ols = model_ols.fit()

ols_summary =  results_ols.summary()
print("OLS summary: \n{}".format(ols_summary))
print("\n\n\n")


## Using Sklearn LinearRegression
lm = LinearRegression()
lm.fit(train_X, train_y)

lm_result = np.mean(cross_val_score(lm, train_X, train_y, cv=6, scoring='neg_mean_absolute_error'))
print("Linear Regressor: {}".format(lm_result))
print("\n\n\n")


############################
# Lasso Regression
############################

## Because data is going to be sparce due to
clm = Lasso()
clm.fit(train_X, train_y)

clm_result = np.mean(cross_val_score(clm, train_X, train_y, cv=6, scoring='neg_mean_absolute_error'))
print("Lasso Regressor: {}".format(clm_result))
print("\n\n\n")

## Aplpy multiple alphas for cross validation of Lasso
alphas = []
errors = []

for i in range(50,110):
    alphas.append(i/2)
    clm = Lasso(alpha=(i/2), normalize=True, copy_X=True)
    clm.fit(train_X, train_y)
    c_v = cross_val_score(clm, train_X, train_y, cv=6, scoring='neg_mean_absolute_error')
    error = np.mean(c_v)
    errors.append(error)

## Plot results
plt.plot(alphas, errors)

## Select best choice of Alppha
for  i, j in enumerate(errors):
    if j == max(errors):
        print("Alpha = {}: min error = {}".format(alphas[i], errors[i]))
        clm = Lasso(alpha=alphas[i], normalize=True, copy_X=True)
        clm.fit(train_X, train_y)
        np.mean(cross_val_score(clm, train_X, train_y, cv=6, scoring='neg_mean_absolute_error'))
        print("\n\n\n")


############################
# Ensamble Models
############################

# Random forest (Tree base model)
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, train_X, train_y, scoring='neg_mean_absolute_error', cv=6))

# Gradient boosted regression tree 
gb = GradientBoostingRegressor(learning_rate=0.05)
gb.fit(train_X, train_y)

# XGBoost 
xgb = XGBRegressor(learning_rate=0.05, n_jobs=4)
xgb.fit(train_X, train_y)


"""Come Back and Displace Feature Importance for Ansamble Models"""
# feat_imp = pd.Series(xgb.booster().get_fscore()).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')

# scores = xgb.feature_importances_
# data =  pd.DataFrame(scores, index=list(train_X.columns))
# data = data.sort_values(by=0, ascending=False)




#############################################################################################
### Tune models using Grid SearchCV or Randomized Grid Search ###
#############################################################################################


# If low on time --> low_oon_time = True
low_on_time = True 


if low_on_time:
    # Randomized Grid Search Cross Validatioon
    parameters_1 = {'n_estimators':range(10,60,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
    rf_gs = RandomizedSearchCV(rf, parameters_1, scoring='neg_mean_absolute_error', cv=3)
    rf_gs.fit(train_X, train_y)
    print("Randomized Grid Search on Random Fores \nScore: {} \nModel: {} \n".format(rf_gs.best_score_, rf_gs.best_estimator_))
    print("\n\n\n")
    
    
    # Randomized Grid Search Cross Validatioon On Gradient Boosting
    parameters_2 = {'n_estimators':range(100,1000,100),'criterion':('friedman_mse','mse','mae'),'max_features':('auto','sqrt','log2')}
    gb_gs = RandomizedSearchCV(gb, parameters_2, scoring='neg_mean_absolute_error', cv=3)
    gb_gs.fit(train_X, train_y)
    print("Randomized Grid Search on Gradient Boosted Regressor \nScore: {} \nModel: {} \n".format(gb_gs.best_score_, gb_gs.best_estimator_))
    print("\n\n\n")
    
    
    # Randomized Grid Search Cross Validatioon On Gradient Boosting
    parameters_3 = {'n_estimators':range(100,1000,100),'criterion':('friedman_mse','mse','mae'),'max_features':('auto','sqrt','log2')}
    xgb_gs = RandomizedSearchCV(xgb, parameters_3, scoring='neg_mean_absolute_error', cv=3)
    xgb_gs.fit(train_X, train_y)
    print("Randomized Grid Search on XGB Regressor \nScore: {} \nModel: {} \n".format(xgb_gs.best_score_, xgb_gs.best_estimator_))
    print("\n\n\n")


else:
    # Grid Search Cross Validatioon
    parameters_1 = {'n_estimators':range(10,60,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
    rf_gs = GridSearchCV(rf, parameters_1, scoring='neg_mean_absolute_error', cv=3)
    rf_gs.fit(train_X, train_y)
    print("Grid Search on Random Fores \nScore: {} \nModel: {} \n".format(rf_gs.best_score_, rf_gs.best_estimator_))
    print("\n\n\n")
    
    
    # Grid Search Cross Validatioon On Gradient Boosting
    parameters_2 = {'n_estimators':range(100,1000,100),'criterion':('friedman_mse','mse','mae'),'max_features':('auto','sqrt','log2')}
    gb_gs = GridSearchCV(gb, parameters_2, scoring='neg_mean_absolute_error', cv=3)
    gb_gs.fit(train_X, train_y)
    print("Grid Search on Gradient Boosted Regressor \nScore: {} \nModel: {} \n".format(gb_gs.best_score_, gb_gs.best_estimator_))
    print("\n\n\n")
    
    
    # Grid Search Cross Validatioon On Gradient Boosting
    parameters_3 = {'n_estimators':range(100,1000,100),'criterion':('friedman_mse','mse','mae'),'max_features':('auto','sqrt','log2')}
    xgb_gs = GridSearchCV(xgb, parameters_3, scoring='neg_mean_absolute_error', cv=3)
    xgb_gs.fit(train_X, train_y)
    print("Grid Search on XGB Regressor \nScore: {} \nModel: {} \n".format(xgb_gs.best_score_, xgb_gs.best_estimator_))
    print("\n\n\n")


"""Come Back and Include SVMs"""
# Support vector regression




#############################################################################################
### Test ensambels ###
#############################################################################################


# Get Model Salary Predictions
lm_pred = lm.predict(val_X)
clm_pred = clm.predict(val_X)

rf_gs_pred = rf_gs.best_estimator_.predict(val_X)
gb_gs_pred = gb_gs.best_estimator_.predict(val_X)
xgb_gs_pred = xgb_gs.best_estimator_.predict(val_X)


# Get Model MAE Scores
lm_error = mean_absolute_error(val_y, lm_pred)
clm_error = mean_absolute_error(val_y, clm_pred)

rf_gs_error = mean_absolute_error(val_y, rf_gs_pred)
gb_gs_error = mean_absolute_error(val_y, gb_gs_pred)
xgb_gs_error = mean_absolute_error(val_y, xgb_gs_pred)


# Gey Model R2 Scores 
lm_r2 = r2_score(val_y, lm_pred)
clm_r2 = r2_score(val_y, clm_pred)

rf_gs_r2 = r2_score(val_y, rf_gs_pred)
gb_gs_r2 = r2_score(val_y, gb_gs_pred)
xgb_gs_r2 = r2_score(val_y, xgb_gs_pred)


# Get Model Variance Scores
lm_var = explained_variance_score(val_y, lm_pred)
clm_var = explained_variance_score(val_y, clm_pred)

rf_gs_var = explained_variance_score(val_y, rf_gs_pred)
gb_gs_var = explained_variance_score(val_y, gb_gs_pred)
xgb_gs_var = explained_variance_score(val_y, xgb_gs_pred)



# Display Model Perfomance
print("Simple Models")
print("################")

model_r2s = [lm_r2, clm_r2, rf_gs_r2, gb_gs_r2, xgb_gs_r2]
model_vars = [lm_var, clm_var, rf_gs_var, gb_gs_var, xgb_gs_var]
model_errors = [lm_error, clm_error, rf_gs_error, gb_gs_error, xgb_gs_error]
model_names = ["Linear Regressor", "Lasso Regressor", "Random Forest", "GB Regressor", "XGB Regressor"]
num_models = range(len(model_names))

for i in num_models:
    name = model_names[i]
    error = np.mean(model_errors[i])
    var = np.mean(model_vars[i])
    r2 = np.mean(model_r2s[i])
    
    print("\n{}".format(name))
    print("MEA: \t{}".format(error))
    print("Var: \t{}".format(var))
    print("R2: \t{}".format(r2))
    
print("\n\n\n")
    


# Display Ensemble Model Performance
print("Ensembled Models")
print("################")

model_preds = [lm_pred, clm_pred, rf_gs_pred, gb_gs_pred, xgb_gs_pred]

for i in num_models[:-1]:
    mp_i = model_preds[i]
    
    for j in num_models[i+1:]:
        mp_j = model_preds[j]
        
        names = [model_names[i], model_names[j]]
        error = np.mean(mean_absolute_error(val_y, (mp_i+mp_j)/2))
        var = np.mean(explained_variance_score(val_y, (mp_i+mp_j)/2))
        r2 = np.mean(r2_score(val_y, (mp_i+mp_j)/2))
        
        print("\n{} & {}".format(names[0], names[1]))
        print("MEA: \t{}".format(error))
        print("Var: \t{}".format(var))
        print("R2: \t{}".format(r2))

print("\n\n\n")

    

# Save best model
pickl = {'model': gb_gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    

model.predict(np.array(list(val_X.iloc[1,:])).reshape(1,-1))[0]
list(val_X.iloc[1,:])