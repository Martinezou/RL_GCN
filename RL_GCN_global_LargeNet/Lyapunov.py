"""Calculate maximum lyapunov exponent of different coupled physical system.
The region below zero is s."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def lyap(cl):
    def diff_chua(u):
        k = 1
        alpha = -0.1
        beta = -1
        gamma = 0.1
        a = 1
        b = -2.5
        H = [1.8348, 10, 10,
             1.1002, 0, 0,
             -0.3254, 1, 0]
        x,y,z = u
        f = [(-k*alpha-k*alpha*b+cl*H[0])*x - 0*k*alpha*a*x**3 + (k*alpha+cl*H[1])*y + cl*H[2]*z,
             (k+cl*H[3])*x+(-k+cl*H[4])*y+(k+cl*H[5])*z,
             cl*H[6]*x +(-k*beta+cl*H[7])*y+(-k*gamma+cl*H[8])*z]
        Df = [[-k*alpha-k*alpha*b+cl*H[0], k*alpha+cl*H[1], 0+cl*H[2]],
              [k+cl*H[3], -k+cl*H[4], k+cl*H[5]],
              [0+cl*H[6], -k*beta+cl*H[7], -k*gamma+cl*H[8]]]
        return np.array(f), np.array(Df)

    def diff_Lorenz(u):
        H = [1, 0, 0,
              0, 0, 0,
              0, 0, 0]
        sigma = 10
        r = 28
        bb = 8 / 3
        x, y, z = u
        f = [sigma * (y - x)+cl*H[0]*x + cl*H[1]*y + cl*H[2]*z,
             r * x + cl*H[3]*x - y+cl*H[4]*y - x * z+ cl*H[5]*z,
             x * y + cl*H[6]*x +cl*H[7]*y +cl*H[8]*z - bb * z]
        Df = [[-sigma+cl*H[0], sigma+cl*H[1], 0+cl*H[2]],
              [r - z+cl*H[3], -1+cl*H[4], -x+cl*H[5]],
              [y+cl*H[6], x+cl*H[7], -bb+cl*H[8]]]
        return np.array(f), np.array(Df)

    def diff_circuit(u):
        H = [1, 0, 0,
             0, 0, 0,
             0, 0, 0]
        a = 1
        b = 3
        c = 1
        d = 5
        r = 0.06
        s = 4.0
        I = 3.2
        x, y, z =u
        f = [y-a*x**3+b*x**2-z+I + cl * H[0] * x + cl * H[1] * y + cl * H[2] * z,
             c-d*x**2-y + cl * H[3] * x + cl * H[4] * y + cl * H[5] * z,
             r*(s*(x+1.6)-z) + cl * H[6] * x + cl * H[7] * y + cl * H[8] * z]
        Df = [[-3*a*(x)**2+2*b*(x) + cl * H[0], 1 + cl * H[1], -1 + cl * H[2]],
              [-2*d*(x) + cl * H[3], -1 + cl * H[4], cl * H[5]],
              [r*s + cl * H[6], 0 + cl * H[7], -r + cl * H[8]]]
        return np.array(f), np.array(Df)

    def diff_rossler(u):
        H = [1, 0, 0,
             0, 0, 0,
             0, 0, 0]
        a = 0.2
        b =0.2
        c= 5.7
        x, y, z =u
        f = [-y-z + cl * H[0] * x + cl * H[1] * y + cl * H[2] * z,
             x+a*y + cl * H[3] * x  + cl * H[4] * y +cl * H[5] * z,
             b+z*(x-c) + cl * H[6] * x + cl * H[7] * y + cl * H[8] * z]
        Df = [[0 + cl * H[0], -1 + cl * H[1], -1 + cl * H[2]],
              [1 + cl * H[3], a + cl * H[4], cl * H[5]],
              [z + cl * H[6], 0 + cl * H[7], x-c + cl * H[8]]]
        return np.array(f), np.array(Df)

    def LEC_system(u):
        #x,y,z = u[:3]
        U = u[3:12].reshape([3,3])
        L = u[12:15]
        f,Df = diff_circuit(u[:3])
        A = U.T.dot(Df.dot(U))
        dL = np.diag(A).copy();
        for i in range(3):
            A[i,i] = 0
            for j in range(i+1,3): A[i,j] = -A[j,i]
        dU = U.dot(A)
        return np.concatenate([f,dU.flatten(),dL])

    u0 = np.ones(3)
    U0 = np.identity(3)
    L0 = np.zeros(3)
    u0 = np.concatenate([u0, U0.flatten(), L0])
    t = np.linspace(0,100,2000)
    u = odeint(lambda u,t:LEC_system(u),u0,t, hmax=0.05)
    L = u[5:,12:15].T/t[5:]
    LE = L[:,-1]
    MLE = np.max(LE)
    return MLE

x = np.arange(-1, 0, 0.01)
positive_list =[]
LElist = []
for i in range(len(x)):
    a = lyap(x[i])
    if a<0:
        positive_list.append(x[i])
    LElist.append(a)
print(positive_list)

plt.plot(x, LElist)
plt.plot(x, list([0 for i in range(len(x))]))
plt.ylabel('maximum lyapunov exponent')
plt.xlabel(r'$\gamma$')
plt.savefig('max_lyap.pdf')
plt.show()