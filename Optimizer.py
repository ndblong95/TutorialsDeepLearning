#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:54:48 2019

@author: dominic
"""
import math as math
import numpy as np
import matplotlib.pyplot as plt
import torch.tensor as tf
from torch.autograd import Variable
import pickle as pickle
import csv
# =============================================================================
# Data
# =============================================================================
x_data = [[2.1,0.1], [4.2,0.8], [3.1,0.9], [3.3,0.2]]
y_data = [0.0, 1.0, 0.0, 1.0]
def readFile(dirData):
    array=[]
    with open(dirData, 'rt') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            array.append(row)
    return array
    csvFile.close()

w=np.array(0.01)
stepsize = 1e-2
e=0.01
fRecur=0.0
fRecur1=0.0
alpha=0.9
beta=0.999
scale=1.0507009873554804934193349852946

def sigmoid(z):
    return 1/(1+np.power(np.e,-z))    

def forward(x):
    return x * w
# Loss function
def loss(x, y):
    y_pred = sigmoid(forward(x))
    return y*np.log10(y_pred) + (1-y)*np.log10(1-y_pred)
# compute gradient
def gradient(x, y):
    return 2 * x * ((x * w) - y)
#backward
#print(backward(1,2))
#def backpropagation(x,y,index):
#    fRecur=momen(x,y,index)
#    fRecur1=velocity(x,y,index)
#    return w- (e*fRecur/((math.sqrt(fRecur1)+stepsize)))

def backpropagation(x,y):
    return w- 0.01*gradient(x,y)
def adaGrad(x, y):
    if fRecur==0.0:
        return np.pow(gradient(x, y),2)
    else:
        return adaGrad(x, y) + np.pow(gradient(x, y),2)

def momen(x,y,index):
    if fRecur==0.0:
        return (1- alpha)*gradient(x, y)
    else:
        return ((alpha*momen(x,y) + (1- alpha)*gradient(x, y)))/(1-pow(alpha,index))

def velocity(x,y,index):
    if fRecur1==0.0:
        return (1- beta)*pow(gradient(x, y),2)
    else:
        return ((beta*velocity(x,y) + (1- beta)*pow(gradient(x, y),2)))/(1-pow(beta,index))

def RMSprop(x,y):
    if fRecur==0.0:
        return 0.1*(gradient(x,y)*gradient(x,y))
    else:
        return 0.9* RMSprop(x,y)+ 0.1*(gradient(x,y)*gradient(x,y))

# =============================================================================
# Activation Function
# =============================================================================
def Relu(x):
    return max(x,0)

def Elu(x):
    if x>0:
        return x
    else:
        return alpha*(pow(np.e,x)-1) #alpha = 1.67326
def Selu(x):
    if x>0:
        return x
    else:
        return scale*Elu(x) #Scale =1.0507009873554804934193349852946
def PReLu(x):
    if x>0:
        return x
    else:
        return alpha*x
def LeakyReLu(x):
    if x>0:
        return x
    else:
        return 0.01*x
def HardTanh(x):
    return max(-1,min(1,x))
def TanH(x):
    z=pow(np.e,x)
    return (z-1/z)/(z+1/z)

dirData= "/home/dominic/Desktop/Trainning/Test/diabetes.csv"
file= readFile(dirData)
data=np.array(file)
data=data[1:].astype(float)
test=data[0,0:7].reshape(-1,1)

#print(loss(test,data[0,8]))

arrW=[]
arrLoss=[]

for layer in np.arange(11):
    print("w=", layer)
    l_sum = 0
    for x_val, y_val in zip(data[:3,0:7], data[:3,7]):
        w=backpropagation(x_val,y_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("\t", l)
    print("MSE=", -l_sum / 3)
    arrW.append(w)
    arrLoss.append(l_sum / 3)
print(arrLoss)
plt.plot(arrLoss)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()