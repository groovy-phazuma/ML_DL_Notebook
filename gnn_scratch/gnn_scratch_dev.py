# -*- coding: utf-8 -*-
"""
Created on 2023-06-01 (Thu) 22:55:51

Scratch GNN

@author: I.Azuma
"""
#%%
import numpy as np

import sys
sys.path.append('/workspace/github/DL-notebook')

from gnn.utils import splittraintest
from gnn.utils import graphreader
from gnn.model import gnn_v1

#%% load dataset
path = '/workspace/github/DL-scratch/gnn/datasets/train/'

dat = splittraintest.split(path,testratio=0.3)
dat.gettraintest()
trainpaths = dat.trainpaths
reader = graphreader.graphreaderbatch(filespath=trainpaths,filelist=True)
adj = reader.adj

#%%
trainset,testset = splittraintest.split(path,testratio=0.3).gettraintest()
trainseteval,testseteval = splittraintest.split(path,testratio=0.3).gettraintest()

batchsize=64
stepinepoch = int(np.ceil(trainset.numexamples/batchsize))
T=2
D=8
net = gnn_v1.GNN(T,D)

#%%
import glob

path = '/workspace/github/DL-scratch/gnn/datasets/train/'
filespaths = glob.glob(path+"//*graph.txt")

adj = np.loadtxt(path,skiprows=1)

#%%
epoch = 100
while trainset.epochcompleted<=epoch :
    adj, nnodes, labels = trainset.nextbatch(batchsize)
    s,_ = net.forward(nnodes,adj)  
    loss = net.loss(s,labels)
    net.backward(loss,labels,upsilon)
    optim.step()

    if  trainset.epochcompleted != epflag :
        print("epoch ",trainset.epochcompleted)
        trainadj, trainnnodes, trainlabels = trainseteval.nextbatch(trainset.numexamples)
        trains,trainout = net.forward(trainnnodes,trainadj)  
        trainloss = np.average(net.loss(trains,trainlabels))
        trainright = np.sum(np.array(trainout)==np.array(trainlabels))
        trainacc = trainright/len(trainout)
        print("train loss : ",trainloss,"train acc :",trainacc)
        trainlosses.append(trainloss)
        trainaccuracies.append(trainacc)
    
        testadj, testnnodes, testlabels = testseteval.nextbatch(testset.numexamples)
        tests,testout = net.forward(testnnodes,testadj)  
        testloss = np.average(net.loss(tests,testlabels))
        testright = np.sum(np.array(testout)==np.array(testlabels))
        testacc = testright/len(testout)
        print("test loss : ",testloss,"test acc :",testacc)
        testlosses.append(testloss)
        testaccuracies.append(testacc)        
        epflag +=1
    step+=1
