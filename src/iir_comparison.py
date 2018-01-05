
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Filtre analogique
wc=100
b, a = sig.butter(2, wc, 'low', analog=True)
Ha=sig.TransferFunction(b,a)
wa, Haj = sig.freqresp(Ha)

# Filtre numérique
Te=0.001

""" Impulse Invariance"""
num=np.array([0.000000000001,0.00931,0.000000000001])
den=np.array([1,-1.8588,0.8681])
H=sig.TransferFunction(num,den,dt=Te)
print(H)

"""Bilinear transform"""
c=wc/np.tan(wc*Te/2)
num=np.array([10000,20000,10000])
den=np.array([(10000-200*c*np.cos(3*np.pi/4)+c*c),(20000-2*c*c),(10000+200*c*np.cos(3*np.pi/4)+c*c)])
H2=sig.TransferFunction(num,den,dt=Te)

# Affichage
w_vect=np.logspace(1,3.49,1000)
w, Hij = sig.dfreqresp(H,w=w_vect*Te)
wb, Hbj = sig.dfreqresp(H2,w=w_vect*Te)
wa, Haj = sig.freqresp(Ha,w=w_vect)


plt.semilogx(wa,20*np.log10(np.abs(Hij)))
plt.semilogx(wa,20*np.log10(np.abs(Hbj)))
plt.semilogx(wa,20*np.log10(np.abs(Haj)))


plt.figure()
plt.semilogx(wa,np.unwrap(np.angle(Hij)))
plt.semilogx(wa,np.unwrap(np.angle(Hbj)))
plt.semilogx(wa,np.angle(Haj))

M=np.matrix([wa,20*np.log10(np.abs(Haj)),20*np.log10(np.abs(Hij)),20*np.log10(np.abs(Hbj))])
M=M.T
np.savetxt("../csv/iir_comp.txt",M,delimiter=',')




plt.show()

