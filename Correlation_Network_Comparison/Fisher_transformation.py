#!/usr/bin/env python3
"""
Created on 2023-12-04 (Mon) 22:28:32

References
- https://github.com/c60evaporator/covariance-visualization
- https://toukeier.hatenablog.com/entry/correlateion-coefficient-difference-test

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn.datasets import load_iris

# %%
def Fisher_transformation(df=None,x='sepal length (cm)',y='petal length (cm)',p=0.05,verbose=True):
    if df is None:
        data = load_iris()
        df = pd.DataFrame(data['data'],columns=data['feature_names'])
    
    n = len(df)
    r = np.corrcoef(df[x], df[y])[0,1]
    z = np.log((1 + r) / (1 - r)) / 2

    eta_min = z - stats.norm.ppf(q=1-p/2, loc=0, scale=1) / np.sqrt(n - 3)
    eta_max = z - stats.norm.ppf(q=p/2, loc=0, scale=1) / np.sqrt(n - 3)

    rho_min = (np.exp(2 * eta_min) - 1) / (np.exp(2 * eta_min) + 1)
    rho_max = (np.exp(2 * eta_max) - 1) / (np.exp(2 * eta_max) + 1)

    if verbose:
        print(r)
        print(f'95% confident interval: {rho_min}〜{rho_max}')

    return z

def Cor_diff_test(df1,df2,x='sepal length (cm)',y='petal length (cm)'):
    z1 = Fisher_transformation(df1,x=x,y=y)
    z2 = Fisher_transformation(df2,x=x,y=y)

    T = (z1 - z2) / np.sqrt((1/(len(df1)-3)) - (1/(len(df2) - 3)))
    p_value = stats.norm.cdf(T,loc=0,scale=1)

    return p_value

# %%
def main():
    data = load_iris()
    df = pd.DataFrame(data['data'],columns=data['feature_names'])

    # Fisher transformation
    Fisher_transformation(df=df,x='sepal length (cm)',y='petal length (cm)',p=0.05,verbose=True)

    # Test for correlation difference
    df1 = df.iloc[0:len(df)//3,:] # 50 samples
    df2 = df.iloc[len(df)//3:,:] # 100 samples
    p = Cor_diff_test(df1,df2,x='sepal length (cm)',y='petal length (cm)')

    print(f'P-value: {p}')

if __name__ == '__main__':
    main()

