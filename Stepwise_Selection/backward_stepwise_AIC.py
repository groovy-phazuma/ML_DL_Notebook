#!/usr/bin/env python3
"""
Created on 2023-12-03 (Sun) 19:27:27

Backward stepwise with monitoring AIC or BIC.
A method that starts with a model that includes all explanatory variables and decreases variables one by one.

@author: I.Azuma
"""
# %%
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

# %%
def AIC(x, y):
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    y_pred = x @ beta
    resid = y - y_pred
    rss = resid.T @ resid
    AIC = len(y) *np.log(rss/len(y)) + (x.shape[1])*2
    return AIC

def BIC(x, y):
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    y_pred = x @ beta
    resid = y - y_pred
    rss = resid.T @ resid
    BIC = len(y) *np.log(rss/len(y)) + (x.shape[1])*np.log(len(y))
    return BIC

def backward_stepwise_result(x, y):
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    return beta

def backward_stepwise(x, y, method='BIC',verbose=False):
    """
    Args:
        x (DataFrame): Dataset for explanatory variables.
        y (DataFrame): Dataset for objective variables.
        method (str, optional): Defaults to 'BIC'.
        verbose (bool, optional): Defaults to False.

    """
    x_tmp = np.concatenate(
        [np.reshape(np.ones(x.shape[0]), (x.shape[0],1)), x], axis=1
    ) 
    included = list([0]) # initial column
    while True:
        changed = False
        excluded = list(set(np.arange(x_tmp.shape[1]))-set(included) )
        result = np.zeros(len(excluded))
        if method == 'AIC':
            base_result = AIC(x_tmp[:, included], y) # calc base AIC
        else:
            base_result = BIC(x_tmp[:, included], y) # cal base BIC
        if verbose:
            print('Baseline model with {}:{:}'.format(method, base_result))
        j = 0
        for new_column in excluded:
            if method == 'AIC':
                result[j] = AIC(x_tmp[:, included + [new_column]], y) # add other column and calc AIC
            else:     
                result[j] = BIC(x_tmp[:, included + [new_column]], y) # add other column and calc BIC
            j += 1
        if result.min() < base_result:
            best_feature = excluded[result.argmin()]
            included.append(best_feature)
            changed = True
            if verbose:
                print('Finally Add {:15} with {} {:}'.format(best_feature, method, result.min()))
        if not changed: #changed=False
            if verbose:
                print('Any variables does not added, stop forward stepwise regression')
            break
                
    # final result
    beta = np.reshape(np.zeros(x_tmp.shape[1]), (x_tmp.shape[1],1))
    beta = backward_stepwise_result(x_tmp[:, included], y)
    included.pop(0) # remove initial column
    included_r = [ i - 1 for i in included]
    included_column = [x.columns.tolist()[idx] for idx in included_r]
    result_min = base_result

    return beta, included_column, result_min

# %%
def main():
    dataset = fetch_california_housing()
    # explanatory variables
    x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # objective variables
    y = pd.Series(dataset.target, name='target')

    print('all variables: AIC = ', AIC(x, y))
    print('all variables: BIC = ', BIC(x, y))
    # forward stepwise
    print("---{}---".format("AIC"))
    beta_AIC, included_column_AIC, result_min_AIC = backward_stepwise(x, y, method='AIC')
    print(beta_AIC)
    print(included_column_AIC)
    print(result_min_AIC)

    # forward stepwise
    print("---{}---".format("BIC"))
    beta_BIC, included_column_BIC, result_min_BIC = backward_stepwise(x, y, method='BIC')
    print(beta_BIC)
    print(included_column_BIC)
    print(result_min_BIC)


if __name__ == '__main__':
    main()