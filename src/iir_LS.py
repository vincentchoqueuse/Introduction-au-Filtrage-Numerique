
import numpy as np
import scipy.signal as sig
import numpy.linalg as lg
import matplotlib.pyplot as plt
from scipy import optimize


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


rranges = (slice(-2, 2, 0.1), slice(-2, 2, 0.1),slice(-2, 2, 0.1), slice(-2, 2, 0.1),slice(-2, 2, 0.1))

resbrute = optimize.brute(cost_function, rranges, args=params, full_output=True,finish=optimize.fmin)
print(resbrute[0])
x0=resbrute[0]



res = minimize(cost_function, x0)
omega=res.x
print(omega)

num,den=extract_num_den(omega)
H=sig.TransferFunction(num,den,dt=Te)



w_vect=np.logspace(1,3.49,100)
wb, Hj = sig.dfreqresp(H,w=w_vect*Te)

t,rit=sig.dimpulse(H,n=200)
plt.figure()
plt.plot(np.ravel(rit))
plt.figure()
plt.semilogx(w_vect,20*np.log10(np.abs(Hj)))

RF=np.zeros((len(w_vect),2))
RF[:,0]=w_vect
RF[:,1]=20*np.log10(np.abs(Hj))
#np.savetxt("../csv/iir_prony_rf.txt",RF,delimiter=',')

plt.show()

