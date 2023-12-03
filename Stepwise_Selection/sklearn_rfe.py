#!/usr/bin/env python3
"""
Created on 2023-12-03 (Sun) 18:25:42

Development for forward-backward stepwise selection

Forward-Backward: Starting from a model that includes all explanatory variables, the variables are increased or decreased one by one.

Backward-Forward: A method that starts with a model that does not include any explanatory variables and increases or decreases variables one by one.

References
- https://www.salesanalytics.co.jp/datascience/datascience145/


@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt

# %% loa data
dataset = fetch_california_housing()
# explanatory variables
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# objective variables
y = pd.Series(dataset.target, name='target')
# concat
df = pd.concat([y,X], axis=1)
print(df)

# display
df.hist(bins=30)
plt.tight_layout()

# correlation
cor = df.corr()
sns.heatmap(cor, annot=True)
plt.show()
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    test_size=0.3,
    random_state=123
)

# multiple regression
model = LinearRegression()
model.fit(X_train, y_train)
print('R2(train):', 
      model.score(X_train, y_train))
print('R2(test):', 
      model.score(X_test, y_test))

# random forest
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
print('R2(train):', model_rf.score(X_train, y_train))
print('R2(test):', model_rf.score(X_test, y_test))

# %%
#rfe = RFE(LinearRegression())
rfe = RFE(RandomForestRegressor())
# feature selection
X_rfe = rfe.fit(X_train,y_train)  
X_selected = X.columns[X_rfe.support_]
print(X_selected)

model_rf.fit(X_train[X_selected], y_train)
print('R2(train):', model_rf.score(X_train[X_selected],y_train))
print('R2(test):', model_rf.score(X_test[X_selected],y_test))

# %% RFE with CV
#rfecv = RFECV(LinearRegression())
rfecv = RFECV(RandomForestRegressor())
# feature selection
X_rfecv = rfecv.fit(X_train,y_train)  
X_selected = X.columns[X_rfecv.support_]
print(X_selected)

model.fit(X_train[X_selected], y_train)
print('R2(train):', model.score(X_train[X_selected],y_train))
print('R2(test):', model.score(X_test[X_selected],y_test))

# %%
