""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  

Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  

We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  

-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
"""

import math
import sys
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
from matplotlib import pyplot as  plt
import pandas as pd
import time

#data prep
test_x, test_y, train_x, train_y = None, None, None, None
permutation = None
author = None
f = 'Data/Istanbul.csv'
datafile = 'Istanbul.csv'
alldata = np.genfromtxt(f, delimiter=",")
# Skip the date column and header row if we're working on Istanbul data
if datafile == "Istanbul.csv":
    alldata = alldata[1:, 1:]
datasize = alldata.shape[0]
cutoff = int(datasize * 0.6)
permutation = np.random.permutation(alldata.shape[0])
col_permutation = np.random.permutation(alldata.shape[1] - 1)
train_data = alldata[permutation[:cutoff], :]
# train_x = train_data[:,:-1]
train_x = train_data[:, col_permutation]
train_y = train_data[:, -1]
test_data = alldata[permutation[cutoff:], :]
# test_x = test_data[:,:-1]
test_x = test_data[:, col_permutation]
test_y = test_data[:, -1]

def LinLearner(train_x,train_y,test_x,test_y):
    print("Lin Reg Learner:")
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())
    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    #print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")
    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    #print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

def DTLearn(train_x, train_y, test_x, test_y):
    print("DT Learner:")
    learner = dt.DTLearner(verbose=True)
    learner.add_evidence(train_x, train_y)
    print(learner.author())
    # evaluate in sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")
    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

def RTLearn(train_x, train_y, test_x, test_y):
    print("RT Learner:")
    learner = rt.RTLearner(verbose=True)
    learner.add_evidence(train_x, train_y)
    print(learner.author())
    # evaluate in sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")
    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

def BagLearn(train_x, train_y, test_x, test_y):
    print("Bag Learner:")
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 20}, bags=1, boost=False, verbose=False)
    learner.add_evidence(train_x, train_y)
    # evaluate in sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")
    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


def InsaneLearn(train_x, train_y, test_x, test_y):
    print("Insane Learner:")
    learner = it.InsaneLearner(verbose=False)  # constructor
    learner.add_evidence(train_x, train_y)  # training step
    # evaluate in sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")
    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


def experiment1():
    sampleRMSE = []
    testRMSE = []
    for x in range(50):
        learner = dt.DTLearner( verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        predS = learner.query(train_x)
        rmseS = math.sqrt(((train_y - predS) ** 2).sum() / train_y.shape[0])
        predT = learner.query(test_x)  # get the predictions
        rmseT = math.sqrt(((test_y - predT) ** 2).sum() / test_y.shape[0])
        sampleRMSE.append(rmseS)
        testRMSE.append(rmseT)
    indexRMSE = list(range(50))
    plt.plot(indexRMSE, sampleRMSE, label="Sample Error")
    plt.plot(indexRMSE, testRMSE, label="Test Error")
    plt.legend()
    plt.savefig("exp1fig1.png")
    #next chart
    plt.plot(indexRMSE[:15], sampleRMSE[:15], label="Sample Error")
    plt.plot(indexRMSE[:15], testRMSE[:15], label="Test Error")
    plt.legend()
    plt.savefig("exp1fig2.png")

    d = {"Leaf Size": indexRMSE, "Sample": sampleRMSE, "Test": testRMSE}
    df = pd.DataFrame(d)
    df.to_csv("Experiment1.csv")

def experiment2():
    sampleRMSE = []
    testRMSE = []
    for x in range(50):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": x}, bags=20, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)  # training step
        predS = learner.query(train_x)
        rmseS = math.sqrt(((train_y - predS) ** 2).sum() / train_y.shape[0])
        predT = learner.query(test_x)  # get the predictions
        rmseT = math.sqrt(((test_y - predT) ** 2).sum() / test_y.shape[0])
        sampleRMSE.append(rmseS)
        testRMSE.append(rmseT)
    indexRMSE = list(range(50))

    plt.plot(indexRMSE, sampleRMSE, label="Sample Error")
    plt.plot(indexRMSE, testRMSE, label="Test Error")
    plt.legend()
    plt.savefig("exp2fig1.png")
    #next plot
    plt.plot(indexRMSE[20:45],sampleRMSE[20:45], label = "Sample Error")
    plt.plot(indexRMSE[20:45],testRMSE[20:45], label = "Test Error")
    plt.legend()
    plt.savefig("exp2fig1.png")

def experiment3():
    # DTLearner
    trainDT = []
    queryDT = []
    for x in range(1, 101, 5):
        start_time = time.time()
        learner = dt.DTLearner(leaf_size=x, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)
        tTime = time.time() - start_time
        start_time2 = time.time()
        pred = learner.query(train_x)
        qTime = time.time() - start_time2
        trainDT.append(tTime)
        queryDT.append(qTime)
    #RTLearner
    trainRT = []
    queryRT = []
    for x in range(1,101,5):
        start_time = time.time()
        learner = rt.RTLearner(leaf_size = x, verbose = False) # constructor
        learner.add_evidence(train_x, train_y)
        tTime = time.time() - start_time
        start_time2 = time.time()
        pred = learner.query(train_x)
        qTime = time.time() - start_time2
        trainRT.append(tTime)
        queryRT.append(qTime)
    indexList = list(range(1,101,5))
    # Train Time
    plt.plot(indexList,trainDT, label = "DTLearner")
    plt.plot(indexList,trainRT, label = "RTLearner")
    plt.legend()
    plt.title("DT vs. RT Training Time")
    plt.savefig("exp3fig1.png")

    # Query Time
    plt.plot(indexList, queryDT, label="DTLearner")
    plt.plot(indexList, queryRT, label="RTLearner")
    plt.legend()
    plt.title("DT vs. RT Query Time")
    plt.savefig("exp3fig2.png")

LinLearner(train_x,train_y,test_x,test_y)
DTLearn(train_x,train_y,test_x,test_y)
RTLearn(train_x, train_y, test_x, test_y)
BagLearn(train_x, train_y, test_x, test_y)
InsaneLearn(train_x,train_y,test_x,test_y)
experiment1()
experiment2()
experiment3()
