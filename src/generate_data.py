import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm


w0=0.7
n_vect=np.arange(-2,50)
input=np.sin(w0*n_vect)*(n_vect>=0)
input2=(n_vect>=0)

# Construction des fonctions de transfert
b, a = sig.butter(2,0.2, 'low')
tf=sig.TransferFunction(b, a, dt=0.1)
tf2=tf.to_zpk()
tf2.poles=np.append(tf2.poles,[-1.4])
tf2.zeros=np.append(tf2.zeros,[0])

# Preparation de l'affichage 3D
N=100
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2,N)

tf_list=[tf,tf2]

for indice in range(len(tf_list)):
    tf_temp=tf_list[indice]

    print("--- Definition du filtre %d ---" % indice)
    print("numerateur=",tf_temp.num)
    print("denominateur=",tf_temp.den)
    gain_tot=np.sum(tf_temp.num)/np.sum(tf_temp.den)
    print("H(1)=%f"%gain_tot)

    print("Gain: %f" %tf_temp.gain)
    print("Zeros:")
    print(tf_temp.zeros)
    print("Poles:")
    print(tf_temp.poles)


    # Module de la fonction de transfert
    X, Y = np.meshgrid(x,y)
    mod_H=np.zeros((N,N))
    for u in range(N):
        for v in range(N):
            z=x[u]+1j*y[v]
            num=np.polyval(tf_temp.num,z)
            den=np.polyval(tf_temp.den,z)
            mod_H[u,v]=np.abs(num)/np.abs(den)

    fig = plt.figure(facecolor='white')
    ax = fig.gca(projection="3d")
    ax.plot_surface(X,Y,mod_H,linewidth=0, norm=colors.LogNorm(vmin=0.001,vmax=2),cmap=cm.Blues,rstride=1,cstride=1)
    ax.view_init(elev=54,azim=45)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.xlabel("Im(z)")
    plt.ylabel("Re(z)")
    plt.savefig("../fig/z_plane_%d.png" %indice)

    # Réponse impulsionnelle
    plt.figure()
    t,yout=sig.dimpulse(tf_temp,n=30)
    plt.stem(t,yout[0])
    np.savetxt("../csv/ri_%d.txt" %indice,yout[0],delimiter=',')

    # Réponse frequentielle
    w, h = sig.dfreqresp(tf_temp,n=200)
    amp=abs(h)
    angles = np.unwrap(np.angle(h))
    M=np.matrix([w,amp,angles]).T

    plt.figure()
    plt.plot(w, amp)

    plt.figure()
    plt.plot(w, angles)
    np.savetxt("../csv/rf_%d.txt" %indice,M,delimiter=',')

    #reponse à une sinusoide
    output=sig.lfilter(tf_temp.num,tf_temp.den,input)
    M=np.array([n_vect,input,output]).T
    np.savetxt("../csv/r_sine_%d.txt" %indice,M,delimiter=',')
    plt.figure()
    plt.plot(n_vect,output)

    #reponse à un echelon unitaire
    output2=sig.lfilter(tf_temp.num,tf_temp.den,input2)
    M=np.array([n_vect,input2,output2]).T
    np.savetxt("../csv/r_step_%d.txt" %indice,M,delimiter=',')
    plt.figure()
    plt.plot(n_vect,output2)


plt.show()


#z,p,k=sig.tf2zpk(b, a)[source]
