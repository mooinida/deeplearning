import numpy as np

def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    for idx in range(x.size):
        #f(x+h)의 값 계산
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)
        #f(x-h)의 값 계산
        x[idx]=tmp_val-h
        fxh2=f(x)

        x[idx]=tmp_val
        grad[idx]=(fxh1-fxh2)/(2*h)
    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad
    return x
def function_2(x):
    return x[0]**2+x[1]**2

x=np.array([-3.0,4.0])
print(gradient_descent(function_2,x,lr=1e-10,step_num=100))