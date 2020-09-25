from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as LinR
import sys,re
import numpy as np
from sklearn.model_selection import cross_val_score


lines=[l.strip().split('\t') for l in open(sys.argv[1]).readlines()[1:46]]

x1=1 # train size
x2=int(sys.argv[2])
X1=[]
X2=[]
Y=[]
for line in lines:
    #print(len(line))
    if not (re.compile("^\s*$").match(line[9]) or re.compile("^\s*$").match(line[x1]) or re.compile("^\s*$").match(line[x2]) or line[9]=="-" or line[x2]=="-"):
        X1.append(float(line[x1]))
        X2.append(float(line[x2]))
        Y.append(float(line[9]))
X=list(zip(X1,X2))
reg = LinR(normalize=True).fit(X, Y)
print(reg.score(X, Y))
print(cross_val_score(reg,X,Y,cv=3,scoring='explained_variance').mean())
print(cross_val_score(reg,X,Y,cv=3,scoring='neg_mean_absolute_error').mean())

