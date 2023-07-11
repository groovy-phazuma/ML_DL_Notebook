# -*- coding: utf-8 -*-
"""
Created on 2023-06-01 (Thu) 22:54:16

[reference]
https://github.com/satrialoka/gnn-from-scratch/blob/master/src/model/gnn.py

@author: I.Azuma
"""
import numpy

class GNN():
    def __init__(self,D,T):
        """
        D : feature vector size
        """
        self.D = D
        self.T = T
    
    def aggregate(self,X,adj):
        """
        Input
            X : feature matrix
            adj : adjacency matrix

        Return
            agg : sum of feature matrix
        """
        agg = np.dot(adj,X)
        return agg
    
    def update(self,W,agg):
        """
        Input
            W : DxD weighted matrix
            adj : adjacency matrix

        Return
            W・agg
        """
        # activation function
        x = np.dot(W,np.transpose(agg))
        x = np.transpose(x)
        return x
    
    def readout(self,X):
        """
        Input :
            X : Feature vectors 
        Return :
            hG : Sum of all feature vectors
        """
        hG = np.sum(X,axis=0)
        return hG
    
    def relu(self,inp):
        """
        Rectifier Linear Unit Function, max(0,inp) 
        Input
            imp : input matrix

        Return
            out : output matrix 
        """
        out = np.maximum(inp,1)
        return out
    
    def s(self,hG,A,b):
        """
        Predictor function with parameter A and b
        Inputs
            hG : last output of aggregation module
            A  : D size parameter
            b  : bias of predictor function
        Return
            s : output of predictor function
        """
        s = np.dot(hG,A)+b
        return s

    def sigmoid(self,s):
        """
        sigmoid activation function
        """
        p = 1/(1+np.exp(-s))
        return p
    
    def output(self,p):
        """
        output the predicted class
        """
        out = np.where((p>0.5),1,0)
        return out
    
    def loss(self,s,y):
        """
        loss function
        Args :
            s   : vector of predictor values
            y   : vector of true class labels
        Return :
            losslist : vector of loss values
        """
        losslist = []
        for i in range (len(s)):
            if np.exp(s[i]) > np.finfo(type(np.exp(s[i]))).max:
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * s[i] #avoid overflow
            else :
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * np.log(1+np.exp(s[i]))
            losslist.append(loss)

        return losslist
    
    def forward(self, nnodes, adj, W = None, A = None, b = None):
        """
        forward method to calculate forward propagation of the nets
        Args :
            nnodes  : number of nodes in the batch
            adj     : adjacency matrix
            W       : parameter matrix W
            A       : parameter vector A
            b       : bias b
        Return : 
            slist       : vector of predictor value 
            output list : vector of predicted class`
        """
        slist = []
        outputlist = []
        X = []
       
        # feature vector definition
        feat =  np.zeros(self.D)
        feat[0] = 1

        self.tempnnodes = nnodes
        self.tempadj = adj

        if np.any(W == None) :
            W = self.W   
        if np.any(A == None) :
            A = self.A
        if b == None :
            b = self.b

        for i in range(adj.shape[0]):
            X.append(np.tile(feat,[nnodes[i],1]))
            for j in range(self.T):
                a = self.aggregate(X[i],adj[i])
                x = self.update(W,a)
                out = self.relu(x)
                X[i] = out
            hG = self.readout(X[i])
            s = self.s(hG,A,b)
            p = self.sigmoid(s)
            output = self.output(p)
            slist.append(s)
            outputlist.append(int(output))
        return slist,outputlist
    
    def backward(self,loss,y,epsilon):
        """
        Backpropagation function to calculate and update 
        the gradient of the neural network
        Args :
            loss    : loss vector
            y       : true class label
            epsilon : small pertubation value for numerical 
                      differentiation 

        """
        tempdLdW = np.zeros((self.D,self.D))
        tempdLdA = np.zeros((self.D))
        tempdLdb = 0
        batchsize = len(loss)
        
        for i in range(self.D):
            for j in range(self.D):
                deltaW = np.zeros((self.D,self.D))
                deltaW[i,j]=epsilon
                Wepsilon = self.W+deltaW
                sep,_ = self.forward(self.tempnnodes,self.tempadj,W=Wepsilon)
                lossep = self.loss(sep,y)
                for k in range(batchsize):
                    tempdLdW[i,j] += (lossep[k] - loss[k])/epsilon
                tempdLdW[i,j] = tempdLdW[i,j]/batchsize

        for i in range(self.D):
            deltaA = np.zeros((self.D))
            deltaA[i] = epsilon
            Aepsilon = self.A + deltaA

            sep,_ = self.forward(self.tempnnodes,self.tempadj,A=Aepsilon)
            lossep = self.loss(sep,y)   
            for j in range(batchsize):
                tempdLdA[i] += (lossep[j] - loss[j])/epsilon
            tempdLdA[i] = tempdLdA[i]/batchsize

        bepsilon = self.b + epsilon
        sep,_ = self.forward(self.tempnnodes,self.tempadj,b=bepsilon)
        lossep = self.loss(sep,y) 
        for i in range(batchsize):
            tempdLdb += (lossep[i] - loss[i])/epsilon
        tempdLdb = tempdLdb/batchsize

        self.dLdW = tempdLdW
        self.dLdA = tempdLdA
        self.dLdb = tempdLdb
