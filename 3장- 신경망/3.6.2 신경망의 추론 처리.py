import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
from 3.6.1 MNIST 데이터셋.py import sigmoid

def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)


img=x_train[0]
label=t_train[0]
print(label)

print(img.shape)
img=img.reshape(28,28)
print(img.shape)

img_show(img)