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

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression, Lasso



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

# Multiple linear regresion

## Using Stats Models' OLS
X_ols = sm.add_constant(X)
model_ols = sm.OLS(y,X_ols)
results_ols = model_ols.fit()

results_ols.summary()


## Using Sklearn LinearRegression
lm = LinearRegression()
lm.fit(train_X, train_y)

np.mean(cross_val_score(lm, train_X, train_y, cv=6, scoring='neg_mean_absolute_error'))


# Lasso Regression

## Because data is going to be sparce due to
clm = Lasso()
clm.fit(train_X, train_y)

np.mean(cross_val_score(clm, train_X, train_y, cv=6, scoring='neg_mean_absolute_error'))

alphas = []
errors = []

for i in range(50,110):
    alphas.append(i/2)
    clm = Lasso(alpha=(i/2), normalize=True, copy_X=True)
    clm.fit(train_X, train_y)
    error = np.mean(cross_val_score(clm,
                            train_X, train_y, cv=6,
                            scoring='neg_mean_absolute_error'))
    errors.append(error)

plt.plot(alphas, errors)

for  i, j in enumerate(errors):
    if j==min(errors):
        print("Alpha = {}: error = {}".format(alphas[i], errors[i]))


# Random forest (Tree base model)
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, train_X, train_y, scoring='neg_mean_absolute_error', cv=6))


# In own time... Gradient boosted tree 
# In own time... Support vector  regression


#############################################################################################
### Tune models using GridsearchCV ###
#############################################################################################

parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=6)
gs.fit(train_X, train_y)

gs.best_score_
gs.best_estimator_



#############################################################################################
### Test ensambels ###
#############################################################################################


