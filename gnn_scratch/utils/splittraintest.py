from utils import dataset
import numpy as np
import glob

class split:
    def __init__(self,path,testratio=0.3):
        self.filespaths = glob.glob(path+"//*graph.txt")
        self.labelspaths = glob.glob(path+"//*label.txt")
        self.filelist=True 
        size = len(self.filespaths)

        self.trainindexend = size - int(testratio*size)
        
    def gettraintest(self):
        self.trainpaths = {'data':self.filespaths[:self.trainindexend],'label':self.labelspaths[:self.trainindexend] }

        self.testpaths  = {'data':self.filespaths[self.trainindexend:],'label':self.labelspaths[self.trainindexend:] }

        traindataset = dataset.graphdataset(self.trainpaths,self.filelist)
        testdataset = dataset.graphdataset(self.testpaths,self.filelist)

        return traindataset,testdataset

