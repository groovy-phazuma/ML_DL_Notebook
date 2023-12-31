# -*- coding: utf-8 -*-
"""
Created on 2023-12-30 (Sat) 17:41:56

@author: I.Azuma
"""
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import mean_squared_error

def plot_scatter(data1,data2,xlabel="True Value",ylabel="Predicted Value",title="donor=32606, cell=HSC, day=all", do_plot=False):
    if type(data1) == pd.DataFrame: data1 = data1.values
    if type(data2) == pd.DataFrame: data2 = data2.values
    """
    corrsum = 0
    for i in range(len(data1)):
        corrsum += np.corrcoef(data1[i], data2[i])[1, 0]
    avg_cor = corrsum / len(data2)
    avg_cor = round(avg_cor,4)
    """

    # flatten
    data1 = list(itertools.chain.from_iterable(data1))
    data2 = list(itertools.chain.from_iterable(data2))

    rmse = round(np.sqrt(mean_squared_error(data1, data2)),4)
    total_cor, pvalue = stats.pearsonr(data1,data2) # correlation and pvalue
    total_cor = round(total_cor,4)

    if do_plot:
        # plot
        fig,ax = plt.subplots()
        plt.scatter(data1,data2,alpha=0.8,s=20)
        plt.text(1.0,0.15,'total_R = {}'.format(str(total_cor)), transform=ax.transAxes, fontsize=13)
        #plt.text(1.0,0.10,'avg_R = {}'.format(str(avg_cor)), transform=ax.transAxes, fontsize=13)
        plt.text(1.0,0.05,'RMSE = {}'.format(str(rmse)), transform=ax.transAxes, fontsize=13)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title(title)
        plt.show()

    return total_cor, rmse