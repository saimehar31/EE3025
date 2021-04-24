
import numpy as np
#If using termux
import subprocess
import shlex
#end if
def cheb(N):
    v = np.array([1,0])
    u = np.array([1])
    if(N==0):
        w = u
    elif N!=1:
        for i in range(1,N):
            p = np.convolve(np.array([2,0]),v)
            m = len(p)
            n = len(u)
            w = p + np.concatenate((np.zeros(m-n),u))
            u = v
            v = w
    elif N==1:
        w = v
    return w