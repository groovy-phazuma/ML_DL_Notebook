# -*- coding: utf-8 -*-
"""
Created on 2024-06-18 (Tue) 22:23:55

Galeano, D., Li, S., Gerstein, M & Paccanaro, A. (2020). 
Predicting the frequencies to drug side effects. Nature Communications.

https://www.nature.com/articles/s41467-020-18305-y
matlab version : https://github.com/paccanarolab/Side-effect-Frequencies

@author: I.Azuma
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class DiegoNMF():
    def __init__(self,num_factors=50,alpha=0.05,tolx=1e-4,max_iter=5000,variance=0.01,verbose=False):
        self.R = None
        self.num_factors=num_factors
        self.alpha=alpha
        self.tolx=tolx
        self.max_iter=max_iter
        self.variance=variance
        self.verbose=verbose
    
    def fix_model(self, mask, intMat, seed=123):
        """perform matrix factorization"""
        self.num_rows, self.num_targets = intMat.shape
        self.R = intMat*mask  # mask

        # initialization
        np.random.seed(seed)
        W0 = np.random.random((self.num_rows, self.num_factors))*np.sqrt(self.variance)
        H0 = np.random.random((self.num_factors, self.num_targets))*np.sqrt(self.variance)
        # normalization
        H0 = (H0.T / np.sqrt(np.sum(H0*H0,axis=1))).T  # (n_factor, n_targets)

        # epsilon based on machine precision
        sqrteps = np.sqrt(np.spacing(1))
        
        # filter for clinical trials values
        CT = self.R > 0
        # filter for unobserved associations
        UN = self.R == 0
        
        self.costs = []
        self.deltas = []
        
        pbar = tqdm(range(self.max_iter))
        last_log = self.calc_loss(W0,H0)
        for i in pbar:
            # update W
            numer = (CT*self.R).dot(H0.T)  # CT*R is same to original R
            denom = (CT*(W0.dot(H0)) + self.alpha*UN*W0.dot(H0)).dot(H0.T) + np.spacing(numer) # matlab eps = numpy spacing
            # W = max(0,W0*(numer/denom))
            W = W0*(numer/denom)
            
            # update H
            numer = W.T.dot((CT*self.R))
            denom = W.T.dot(CT*(W.dot(H0)) + self.alpha*UN*W.dot(H0)) + np.spacing(numer)
            # H = max(0,H0*(numer/denom))
            H = H0*(numer/denom)
            
            # compute cost
            cost = 0.5*(np.linalg.norm(CT*(self.R-W.dot(H)),ord="fro")**2) + 0.5*self.alpha*(np.linalg.norm(UN*(self.R-W.dot(H)),ord="fro")**2)
            self.costs.append(cost)
            
            # get norm of difference and max change in factors
            dw = np.amax(abs(W-W0) / (sqrteps+np.amax(abs(W0))))
            dh = np.amax(abs(H-H0) / (sqrteps+np.amax(abs(H0))))
            delta = max(dw,dh)
            self.deltas.append(delta)
            
            curr_log = self.calc_loss(W,H)
            loss = (curr_log-last_log)/abs(last_log)
            last_log = curr_log
            
            pbar.set_postfix(DELTA=delta,LOSS=loss)
            
            if i > 1:
                if delta <= self.tolx:
                    if self.verbose:
                        print("---completed---")
                        print("iter :",i,"delta :",delta)
                    break
            """
            if i > 1:
                if loss <= self.tolx:
                    if self.verbose:
                        print("---completed---")
                        print("iter :",i,"delta :",delta)
                    break
            """
            #if self.verbose:
                #print("iter :",i,"delta :",delta)
            W0 = W
            H0 = (H.T / np.sqrt(np.sum(H*H,axis=1))).T
        # return
        self.W = W0
        self.H = H0
    
    def calc_loss(self,W,H):
        loss = 0
        tmp = self.R - np.dot(W,H)
        loss -= np.sum(tmp*tmp)
        return loss
    
    def evaluation(self,test_data, test_label):
        ii, jj = test_data[:,0], test_data[:,1]
        self.scores = np.sum(self.W[ii,:]*self.H.T[jj,:], axis=1) # note that H is transposed
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
    
    def summarize(self,ntop=200):
        # plot the delta and the loss distribution
        use_delta = self.deltas[ntop:]
        x = [i+ntop for i in range(len(use_delta))]
        plt.plot(x,use_delta)
        plt.xlabel("iterations")
        plt.ylabel("delta")
        plt.show()
        
        use_cost = self.costs[ntop:]
        plt.plot(x,use_cost)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.show()
        
        # plot original matrix and completed one
        sns.heatmap(self.R)
        plt.title("original R")
        plt.show()
        
        sns.heatmap(self.W.dot(self.H))
        plt.title("R' ~ W * H")
        plt.show()

def main(path_dataset):
    u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
    val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values = ru.load_data(path_dataset= path_dataset, seed=1234, verbose=True)

    X = np.array(rating_mx_train.todense())
    mask = np.zeros(X.shape)
    mask[u_train_idx,v_train_idx]=1  # binary mask

    # Non-negative matrix factorization
    NMF = nmf.DiegoNMF(num_factors=50,alpha=0.05,tolx=1e-2,
                    max_iter=100,variance=0.01,verbose=True)
    NMF.fix_model(mask=mask, intMat=X)
    output = NMF.W.dot(NMF.H)

    pred_labels = output[u_test_idx,v_test_idx]
    pu.plot_scatter([test_labels],[pred_labels],do_plot=True,title="Non-negative Matrix Factorization")

    """
    ---Completed---
    iter : 69 delta : 0.009639518942249781

    total_R = 0.0093
    RMSE = 46.4286
    """

if __name__ == '__main__':
    BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Others/github/ML_DL_Notebook'
    import numpy as np
    import sys
    sys.path.append(BASE_DIR)
    sys.path.append(BASE_DIR+'/Recommendation')

    from MF import nmf
    import recomm_utils as ru
    from _utils import plot_utils as pu

    #  Load dataset
    path_dataset = BASE_DIR+'/_datasource/yahoo_music/training_test_dataset.mat'

    main(path_dataset)