
import numpy as np
import scipy.signal as sig
import numpy.linalg as lg
import matplotlib.pyplot as plt


from scipy.optimize import minimize


# Filtre analogique
wa=100
Te=0.001
w=100*Te
N=200

n_vect=np.arange(N)
ri=np.sin(w*(n_vect-N/2))/(np.pi*(n_vect-N/2))
ri[np.floor(N/2)]=(w/np.pi)

M=3
L=3

def extract_num_den(omega):
    den=np.zeros(L)
    den[0]=1
    den[1:]=omega[:L-1]
    num=omega[L-1:]
    return num,den


def cost_function(x0):
    num,den=extract_num_den(x0)
    H=sig.TransferFunction(num,den,dt=Te)
    t,rit=sig.dimpulse(H,n=200)
    error=np.sum(np.abs(ri-rit)**2)
    print(error)
    return error


x0=np.array([-1.85879,0.86812,0.00233,0.00466,0.00233])
num,den=extract_num_den(x0)
H=sig.TransferFunction(num,den,dt=Te)
print(H)
t,rit=sig.dimpulse(H)
print(rit)
plt.plot(np.ravel(rit))
n=input("truc")


res = minimize(cost_function, x0, method='Nelder-Mead', tol=1e-6)
omega=res.x
print(omega)

num,den=extract_num_den(omega)
H=sig.TransferFunction(num,den,dt=Te)

w_vect=np.logspace(1,3.49,100)
wb, Hj = sig.dfreqresp(H,w=w_vect*Te)

plt.plot(ri)
plt.figure()
plt.semilogx(w_vect,20*np.log10(np.abs(Hj)))

RF=np.zeros((len(w_vect),2))
RF[:,0]=w_vect
RF[:,1]=20*np.log10(np.abs(Hj))
#np.savetxt("../csv/iir_prony_rf.txt",RF,delimiter=',')

plt.show()

