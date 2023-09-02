import numpy as np

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y))/batch_size

t=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.arange(10)
print(y)