import numpy as np

def cross_entrophy_error(y,t):
    delta=1e-7 # 마이너스 무한대가 발생하지 않게 하기 위해.
    return -np.sum(t*np.log(y+delta))

t=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y=[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entrophy_error(np.array(y),np.array(t)))

y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(cross_entrophy_error(np.array(y),np.array(t)))