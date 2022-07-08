import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

k1 = 0.04
k2 = 3 * 10**7
k3 = 10**4

c = [0.5, 1.5]
b = [-0.5, 1.5]
a = [[0.5, 0],
     [-0.5, 2]]

def first_step(y, t):
    dt = t[1] - t[0]
    return (1 + dt*k3*y[1]*y[2] - dt*k1*y[0] - y[0],
            0 + dt*k1*y[0] - dt*k3*y[1]*y[2] - dt*k2*y[1]**2 - y[1],
            0 + dt*k2*y[1]**2 - y[2])

def f(y):
    return np.array([-k1*y[0] + k3*y[1]*y[2],
                     k1*y[0] - k3*y[1]*y[2] - k2*y[1]**2,
                     k2*y[1]**2])


def rungekutta(f, y0, k0, t, args=()):  # метод Рунге-Кутты 4-го порядка
    n = len(t)
    y = np.zeros((n, len(y0)))
    k_1 = np.zeros((n, len(y0)))
    k_2 = np.zeros((n, len(y0)))
    k_1[0] = k0
    k_2[0] = k0
    y[0] = y0
    for i in range(n - 1):
        dt = t[i+1] - t[i]
        k_1[i] = f(y[i] + dt*a[0][0]*k_1[i-1] + dt*a[0][1]*k_2[i-1])
        k_2[i] = f(y[i] + dt*a[1][0]*k_1[i-1] + dt*a[1][1]*k_2[i-1])
        y[i+1] = y[i] + dt*(b[0]*k_1[i] + b[1]*k_2[i])
    return y



def main():
    y0_ = (1, 0, 0)
    t = np.linspace(0, 0.3, 3000)
    
    y1, y2, y3 = fsolve(first_step, y0_, t)
    y0 = np.array([y1, y2, y3])
    k0 = [y0[0]-y0_[0], y0[1]-y0_[1], y0[2]-y0_[2]]
    
    y = rungekutta(f, y0, k0, t)
    
    y1 = []
    y2 = []
    y3 = []
    for i in range(len(y)):
        y1.append(y[i][0])
        y2.append(y[i][1])
        y3.append(y[i][2])
        
    
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(131)
    ax.plot(t, y1)
    ax.set_xlabel('t(c)')
    ax.set_ylabel('y1')
    ax = fig.add_subplot(132)
    ax.plot(t, y2)
    ax.set_xlabel('t(c)')
    ax.set_ylabel('y2')
    ax = fig.add_subplot(133)
    ax.plot(t, y3)
    ax.set_xlabel('t(c)')
    ax.set_ylabel('y3')
    
    
if __name__ == '__main__':
    main()