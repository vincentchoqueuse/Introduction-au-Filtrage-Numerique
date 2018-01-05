
import numpy as np
import scipy.signal as sig
import numpy.linalg as lg
import matplotlib.pyplot as plt

# Filtre analogique
wa=100
Te=0.001
w=100*Te
N=200



n_vect=np.arange(N)
ri=np.sin(w*(n_vect-N/2))/(np.pi*(n_vect-N/2))
ri[np.floor(N/2)]=(w/np.pi)

M=6
L=6

H_tot=np.matrix(np.zeros((N,L)))
for start in range(1,L+1):
    H_tot[start:,start-1]=np.matrix(ri[:-start]).T

print(H_tot)
H_0=H_tot[:M+1,:]
H=-H_tot[M+1:,:]
h=np.matrix(ri[:M+1]).T
x=np.matrix(ri[M+1:]).T

a=lg.pinv(H)*x
b=h+H_0*a

num=np.ravel(b)
den=np.zeros(len(a)+1)
den[0]=1
den[1:]=np.ravel(a)
w_vect=np.logspace(1,3.49,100)
H=sig.TransferFunction(num,den,dt=Te)
wb, Hj = sig.dfreqresp(H,w=w_vect*Te)

plt.plot(ri)
plt.figure()
plt.plot(w_vect,20*np.log10(np.abs(Hj)))

RF=np.zeros((len(w_vect),2))
RF[:,0]=w_vect
RF[:,1]=20*np.log10(np.abs(Hj))
np.savetxt("../csv/iir_prony_rf.txt",RF,delimiter=',')

plt.show()

