import numpy as np
import matplotlib.pyplot as plt
#If using termux
import subprocess
import shlex
#end if

N = 6
n = np.arange(N)
fn=(-1/2)**n
hn1=np.pad(fn, (0,2), 'constant', constant_values=(0))
hn2=np.pad(fn, (2,0), 'constant', constant_values=(0))
h = hn1+hn2
xtemp=np.array([1.0,2.0,3.0,4.0,2.0,1.0])
x=np.pad(xtemp, (0,8), 'constant', constant_values=(0))

def fft_matrix(n):
	fft_mat = np.zeros((n,n),dtype=np.complex128)
	for i in range(n):
		for j in range(n):
				fft_mat[i][j] = np.exp(-2j*np.pi*i*j/n)
	return fft_mat

def fft(x):
	n = len(x)
	F = fft_matrix(n)
	return np.real(F@x)


def ifft_matrix(n):
	ifft_mat = np.zeros((n,n),dtype=np.complex128)
	for i in range(n):
		for j in range(n):
				ifft_mat[i][j] = (1/n)*np.exp(2j*np.pi*i*j/n)
	return ifft_mat

def ifft(x):
	n = len(x)
	F = ifft_matrix(n)
	return (F@x)

X = np.fft.fft(x,N)
H = np.fft.fft(h,N)
Y = X*H
y = np.fft.ifft(Y)
y = np.real(y)

print('Values of y(n) that is not calculated using inbuilt command is :',np.real(ifft(Y)))


plt.figure(1)
plt.stem(range(0,N),y,use_line_collection=True)
plt.title('Plotting the output $y(n)$')
plt.xlabel('$n$')
plt.ylabel('$y(n)$')
plt.grid()
#If using termux
#plt.savefig('../figs/y_n.eps')
#plt.savefig('../figs/y_n.pdf')
#subprocess.run(shlex.split("termux-open ../figs/y_n.pdf"))


plt.figure(2,figsize=(9,9))
plt.subplot(2,1,1)
plt.stem(np.abs(Y),use_line_collection=True)
plt.title(r'$|Y(k)|$')
plt.xlabel('$n$')
plt.ylabel(r'$|Y(k)|$')
plt.grid()

plt.subplot(2,1,2)
plt.stem(np.angle(Y),use_line_collection=True)
plt.title(r'$\angle{Y(k)}$')
plt.xlabel('$n$')
plt.ylabel(r'$\angle{Y(k)}$')
plt.grid()
#If using termux
#plt.savefig('../figs/Y_K.eps')
#plt.savefig('../figs/Y_K.pdf')
#subprocess.run(shlex.split("termux-open ../figs/Y_K.pdf"))


plt.figure(3,figsize=(9,7.5))
plt.subplot(2,2,1)
plt.stem(np.abs(x),use_line_collection=True)
plt.title(r'$|x(n)|$')
plt.grid()

plt.subplot(2,2,2)
plt.stem(np.angle(x),use_line_collection=True)
plt.title(r'$\angle{x(n)}$')
plt.grid()

plt.subplot(2,2,3)
plt.stem(np.abs(h),use_line_collection=True)
plt.title(r'$|h(n)|$')
plt.grid()

plt.subplot(2,2,4)
plt.stem(np.angle(h),use_line_collection=True)
plt.title(r'$\angle{h(n)}$')
plt.grid()
#If using termux
#plt.savefig('../figs/x_nh_n.eps')
#plt.savefig('../figs/x_nh_n.pdf')
#subprocess.run(shlex.split("termux-open ../figs/x_nh_n.pdf"))


plt.figure(4,figsize=(9,9))
plt.subplot(2,1,1)
plt.stem(np.abs(X),use_line_collection=True)
plt.title(r'$|X(k)|$')
plt.grid()

plt.subplot(2,1,2)
plt.stem(np.angle(X),use_line_collection=True)
plt.title(r'$\angle{X(k)}$')
plt.grid()
#If using termux
#plt.savefig('../figs/X_K.eps')
#plt.savefig('../figs/X_K.pdf')
#subprocess.run(shlex.split("termux-open ../figs/X_K.pdf"))

plt.figure(5,figsize=(9,9))
plt.subplot(2,1,1)
plt.stem(np.abs(H),use_line_collection=True)
plt.title(r'$|H(k)|$')
plt.grid()

plt.subplot(2,1,2)
plt.stem(np.angle(H),use_line_collection=True)
plt.title(r'$\angle{H(k)}$')
plt.grid()
#If using termux
#plt.savefig('../figs/H_K.eps')
#plt.savefig('../figs/H_K.pdf')
#subprocess.run(shlex.split("termux-open ../figs/H_K.pdf"))

plt.figure(6)
plt.stem(range(0,N),np.real(ifft(Y)),use_line_collection=True)
plt.title(r'$|y(n)|$')
plt.grid()
#If using termux
#plt.savefig('../figs/y_n_1.eps')
#plt.savefig('../figs/y_n_1.pdf')
#subprocess.run(shlex.split("termux-open ../figs/y_n_1.pdf"))



plt.show()

