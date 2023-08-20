import numpy as np
def relu(x):
    return np.maximum(0,x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x-=np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def init_network():
    network={}
    network['W1']=np.array([[0,0,0,0,1],[-1,0,0,0,0],[0,-1,0,0,0],[0,0,-1,0,0],[0,0,0,-1,0]])
    network['b1']=np.array([5,4,3,2,1])
    network['W2']=np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]])
    network['b2']=np.array([-1,-2,0,0,0])

    return network

def forward(network, x):
    W1,W2=network['W1'],network['W2']
    b1,b2=network['b1'],network['b2']

    a1=np.dot(x,W1)+b1
    z1=relu(a1)
    a2=np.dot(z1,W2)+b2
    y=softmax(a2)

    return y

network=init_network()
x=np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]])
y=forward(network,x)
print(y)