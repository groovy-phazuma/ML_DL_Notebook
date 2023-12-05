#!/usr/bin/env python3
"""
Created on 2023-12-05 (Tue) 22:40:36

permutation test

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

import sys
sys.path.append('C:\github\ML_DL_Notebook\Correlation_Comparison')
import Fisher_transformation as ft

# %%
def permutation_test(df1,df2,x='sepal length (cm)',y='petal length (cm)',n_perm=1000,alternative="less"):
    if alternative in ['less','greater']:
        pass
    else:
        raise ValueError("!! Inappropriate alternative type !!")
    original_t,original_p = ft.Cor_diff_test(df1,df2,x=x,y=y,verbose=False)

    n1 = len(df1)
    concat_df = pd.concat([df1,df2])

    # permutation
    perm_res = [original_t]
    for i in tqdm(range(n_perm)):
        shuffle_df = concat_df.sample(frac=1, random_state=i)
        u_df1 = shuffle_df.iloc[0:n1,:]
        u_df2 = shuffle_df.iloc[n1:,:]
        ut,up = ft.Cor_diff_test(u_df1,u_df2,x=x,y=y,verbose=False)
        perm_res.append(ut)
    
    # calc p-value
    if alternative == "less":
        perm_res = sorted(perm_res)
    else:
         perm_res = sorted(perm_res,reverse=True)

    original_idx = perm_res.index(original_t)
    perm_p = original_idx / n_perm

    # visualization
    fig,ax = plt.subplots(figsize=(6,4))
    plt.hist(perm_res,bins=int(n_perm/10),alpha=0.9)
    plt.vlines(x=original_t,ymin=0,ymax=10,color="red",ls="dashed",linewidth=2)

    plt.xlabel('statistics value')
    plt.ylabel('frequency')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.show()

    return perm_p

# %%
def main():
    data = load_iris()
    df = pd.DataFrame(data['data'],columns=data['feature_names'])

    # 1. 50 samples vs 100 samples
    df1 = df.iloc[0:50,:] # 50 samples
    df2 = df.iloc[50:,:] # 100 samples
    perm_p = permutation_test(df1,df2,x='sepal length (cm)',y='petal length (cm)',n_perm=1000,alternative="less")
    print(f'P-value: {p}')

    # 2. 100 samples vs 50 sample
    df1 = df.iloc[0:100,:] # 100 samples
    df2 = df.iloc[100:,:] # 50 samples
    perm_p = permutation_test(df1,df2,x='sepal length (cm)',y='petal length (cm)',n_perm=1000,alternative="less")
    print(f'P-value: {p}')

if __name__ == '__main__':
    main()



# %%
