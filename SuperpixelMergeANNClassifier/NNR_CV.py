"""
Train neural network classifier
Devendra - 2014CSB1010
"""

import os, struct
from array import array as pyarray
import math
from numpy import *
from pylab import *
import numpy as np
import sys
from csv import reader
import re
from collections import Counter
from sklearn.externals import joblib
from numpy import genfromtxt

from sklearn.neural_network import MLPClassifier

#from sklearn.metrics import classification_report,confusion_matrix

np.set_printoptions(threshold='nan')

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# pri is used to control print statements used for debugging in code.
# pri = 1 prints degree-rank graph.
# pri = 2 also prints values of predicted/actual rank.
pri=1

#print list vertically
def printV(lst):
    for x in lst:
        print x;

def doShuffle(degree,rank):
	
	# Shuffle degree and rank array together
	shuf = np.arange(0,len(degree));
	np.random.shuffle(shuf);

	comb=zip(shuf,degree);
	comb=sorted(comb);

	degree=[x[1] for x in comb];

	comb=zip(shuf,rank);
	comb=sorted(comb);

	rank=[x[1] for x in comb];

	
	degree = array(degree);
	rank = array(rank);
	
	return degree,rank
	
# read degree,rank from file    
def load_all(filename):
    
    degree=[]; rank=[]; freq=[];
    with open(filename) as afile:
        r = reader(afile)
        c=0
        cRank=1;
        for line in r:
            # c+=1;
            # if (c<2):
                # continue;
            
            degree.append( map(float,line));
                        
            #freq.append(float(inp[2]));
            
    return array(degree);

	
	
#Data = genfromtxt('output.txt', delimiter=',');
Data = load_all('output.txt')
#Data = Data[0:100,:];

rank = (Data[:,0]).astype(int);
degree = Data[:,1:];
Data = [];
#printV (zip(degree,rank));

print degree.shape, rank.shape

N = len(degree)


degree, rank  = doShuffle(degree,rank);

#rank = rank.reshape(-1,1);

# split data into training and testing instances
splitRatio=0.9 # splitRatio determines how many instances are used for training and testing. eg: 0.2 means 20% train, 80% test
spl= int(splitRatio*N); #split location for train-test
print "Train : ",int(splitRatio*N),"\t Test: ",int((1.0-splitRatio)*N)

trI=array(degree[:spl]); trL=array(rank[:spl]); # trI - training instances, trL - training labels
teI=array(degree[spl:]); teL=array(rank[spl:]); # teI - testing instances, teL - testing labels

trI=trI.astype('float'); teI=teI.astype('float');

# set parameters of neural network classifier model
nn = MLPClassifier(hidden_layer_sizes=(400,200), activation='relu', solver='adam', alpha=0.0001, batch_size=5, learning_rate='constant', learning_rate_init=0.001,  max_iter=100, shuffle=False, random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.4, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1)
	

print trI.shape, trL.shape;

# train NN regression model
nn.fit(trI,trL);    

# test model to get accuracy
res=nn.score(teI,teL);

fil = open('r1.txt','w');
print 'Accuracy measure: ', res
fil.write(str(res));

yres = nn.predict(teI);

# save trained model
joblib.dump(nn, 'ANN.pkl') 
