
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Filtre analogique
wa=100
Te=0.001
w=100*Te

M=200

window_vect=["None","hamming","blackman"]

n_vect=np.arange(M)
h=np.sin(w*(n_vect-M/2))/(np.pi*(n_vect-M/2))
h[np.floor(M/2)]=(w/np.pi)
w_vect=np.logspace(1,3.49,1000)

RI=np.zeros((M,4))
RF=np.zeros((len(w_vect),4))

RI[:,0]=n_vect
RF[:,0]=w_vect
index=1
for window in window_vect:

    if window=="None":
        b=h
    if window=="hamming":
        b=sig.hamming(M)*h
    if window=="blackman":
        b=sig.blackman(M)*h

    H=sig.TransferFunction(b,1,dt=Te)
    wb, Hbj = sig.dfreqresp(H,w=w_vect*Te)

    RI[:,index]=b
    RF[:,index]=20*np.log10(np.abs(Hbj))
    index=index+1


np.savetxt("../csv/fir_w_ri.txt",RI,delimiter=',')
np.savetxt("../csv/fir_w_rf.txt",RF,delimiter=',')


plt.figure()
plt.plot(n_vect,RI[:,1:])
plt.figure()
plt.semilogx(w_vect,RF[:,1:])


plt.show()

