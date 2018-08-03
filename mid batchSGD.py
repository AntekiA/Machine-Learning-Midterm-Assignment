import numpy as np
import math
import random
import matplotlib.pyplot as plt
rate = 0.2  # learning rate
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

print("x: ", x, np.shape(x))
print("y: ", y, np.shape(y))
print("w: ", w, np.shape(w))


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


i = 0
all_loss = []
all_step = []
for step in range(1, 10):
    loss = 0
    batchy = np.mat(y[0, i:i+4])
    batchx = x[i: i+4]
    f = dw(batchy, w, batchx)
    w = w - rate * f
    loss = calc_loss(y, w, x)
    all_loss.append(loss)
    all_step.append(step)
    plt.plot(all_step, all_loss, color='orange')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("BSG")
    if step % 1 == 0:
        print("step: ", step, " loss: ", loss, " w: ", w)
    i = i + 4
plt.show()
