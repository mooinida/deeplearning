import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200,threshold=1000)
(x_train, t_train),(x_test,t_test)=load_mnist(flatten=False,normalize=False)



img=x_train[0][0]
img2=255-x_train[0][0]
img3=x_train[0][0].T

plt.imshow(img,cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()
plt.imshow(img2,cmap=plt.cm.gray)
plt.xticks()
plt.yticks()
plt.show()
plt.imshow(img3,cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()