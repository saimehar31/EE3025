import numpy as np

def add(x,y):
    m = len(x)
    n = len(y)
    if(m==n):
        return x+y
    elif m>n:
        return x + np.concatenate((np.zeros(m-n),y))
    else:
        return y + np.concatenate((np.zeros(n-m),x))