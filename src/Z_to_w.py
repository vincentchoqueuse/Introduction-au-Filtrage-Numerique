import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm

# Coef
b, a = sig.butter(2,0.2, 'low')
w, h = sig.freqz(b, a)


N=100
x = np.linspace(-1.2, 1.2, N)
y = np.linspace(-1.2, 1.2,N)
mod_H=np.zeros((N,N))
for u in range(N):
    for v in range(N):
        z=x[u]+1j*y[v]
        num=np.polyval(b,z)
        den=np.polyval(a,z)
        mod_H[u,v]=np.abs(num)/np.abs(den)

X, Y = np.meshgrid(x,y)
fig = plt.figure(facecolor='white')
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,mod_H,linewidth=0, norm=colors.LogNorm(vmin=0.001,vmax=2),cmap=cm.binary,rstride=1,cstride=1)
ax.view_init(elev=33,azim=55)
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.xlabel("Im(z)")
plt.ylabel("Re(z)")
plt.savefig('../fig/z_plane.png')



"""
plt.figure
w_vect=np.linspace(0,np.pi,N)
print(a)
print(b)
print(np.roots(b))
print(np.roots(a))
h2=np.zeros(N)
indice=0
for w in w_vect:
    z=np.exp(1j*w)
    num=np.polyval(b,z)
    den=np.polyval(a,z)
    h2[indice]=np.abs(num)/np.abs(den)
    indice=indice+1


plt.figure()
plt.plot(w_vect,h2)
M=np.matrix([w_vect,h2]).T
np.savetxt("../csv/f_abs.txt",M,delimiter=',')
"""

w_vect, h = sig.freqz(b, a)
abs_vect=np.abs(h)
angle_vect=np.angle(h)
plt.figure()
M=np.matrix([w_vect,abs_vect,angle_vect]).T
np.savetxt("../csv/f_abs_angle.txt",M,delimiter=',')

plt.show()


#z,p,k=sig.tf2zpk(b, a)[source]
