import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200,threshold=1000)
x=np.arange(25).reshape(5,5)
print(x)
plt.imshow(x,cmap=plt.cm.gray)
plt.title("1st")
plt.xticks([])
plt.yticks([])
plt.show()

plt.imshow(x.T ,cmap=plt.cm.gray)
plt.title("2nd")
plt.xticks([])
plt.yticks([])
plt.show()

plt.imshow(25-x,cmap=plt.cm.gray)
plt.title("3rd")
plt.xticks([])
plt.yticks([])
plt.show()

y=np.zeros((5,5))
y[2,:]=24
plt.imshow(y,cmap=plt.cm.gray)
plt.title("4th")
plt.xticks([])
plt.yticks([])
plt.show()

y=np.zeros((5,5))
y[:,2]=24
plt.imshow(y,cmap=plt.cm.gray)
plt.title("5th")
plt.xticks([])
plt.yticks([])
plt.show()

y=np.zeros((5,5))

y[:,2]=12
y[2,:]=12
y[2,2]=24
plt.imshow(y,cmap=plt.cm.gray)
plt.title("6th")
plt.xticks([])
plt.yticks([])
plt.show()


