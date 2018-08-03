import numpy as np
import math
import random
import matplotlib.pyplot as plt
ramuda = 1.0  # hyper parameter

# data set
n = 40
np.random.seed(5)
omega = np.random.randn(1, 1)
random.seed(5)
noise = 0.8 * np.random.randn(n, 1)
random.seed(5)
x = np.random.randn(n, 2)
y = 2 * (omega * x[:, 0] + x[:, 1] + noise.T > 0) - 1
w = np.mat('10; 10')


def calc_loss(y, w, x):
    error = 0
    two = np.dot(w.T, w)
    for i in range(0, len(y.T)):
        one = (-1) * y[0, i] * np.dot(x[i], w)
        error = error + math.log(1 + math.exp(one[0, 0]))

    return error + ramuda * two[0, 0]


def dw(y, w, x):
    n = np.mat('0; 0')
    for j in range(0, len(y.T)):
        k = (-1) * y[0, j] * np.dot(x[j], w)
        l = math.exp(k[0, 0])
        m = np.mat(x[j])
        n = n + y[0, j] * m.T * l/(1 + l)
    o = n + 2 * ramuda * w
    return o


def hesse(y, w, x):
    H = np.mat('0 ,0; 0, 0', dtype=float)
    h11 = 0
    h22 = 0
    h12 = 0
    for i in range(0,len(y.T)):
        a = (-1) * y[0, i] * np.dot(x[i], w)
        b = math.exp(a[0, 0])
        c = b/(1 + b)
        d = np.mat(x[i])
        e11 = (y[0, i] * d[0, 0])**2
        e22 = (y[0, i] * d[0, 1])**2
        e12 = ((y[0, i])**2) * d[0, 0] * d[0, 1]
        h11 = h11 + (c - (b**2)/((1 + b)**2)) * e11
        h22 = h22 + (c - (b**2)/((1 + b)**2)) * e22
        h12 = h12 + (c - (b**2)/((1 + b)**2)) * e12
    H[0, 0] = h11 + 2 * ramuda
    H[1, 1] = h22 + 2 * ramuda
    H[0, 1] = H[1, 0] = h12
    return H


all_loss = []
all_step = []
for step in range(1, 10):
    loss = 0
    H = hesse(y, w, x)
    f = dw(y, w, x)
    loss = calc_loss(y, w, x)
    w = w - np.dot(H.I, f)
    all_loss.append(loss)
    all_step.append(step)
    plt.plot(all_step, all_loss, color='orange')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("neuton")
    if step % 1 == 0:
        print("step: ", step, " loss: ", loss, " w: ", w)
plt.show()


