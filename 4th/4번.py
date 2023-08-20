import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train,t_train),(x_test,t_test)=load_mnist(flatten=False,normalize=False)

n=2
my=[]
for i in range(len(x_train)):
        if t_train[i]==n:
            my.append(i)
        
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[my[i]][0],cmap=plt.cm.binary)
    plt.xlabel(t_train[my[i]])
plt.show()

