import numpy as np
import matplotlib.pyplot as plt

# Objective function: f(x) + ramda*norm1(x)
def obj(A, x, b, ramda):
    assert (np.size(x, 0) == np.size(A, 1) and np.size(A, 0) == np.size(b, 0) and \
            np.size(x, 1) == np.size(b, 1) == 1 and np.isscalar(ramda))
    return f(A, x, b) + ramda * np.sum(np.abs(x))


# f(x)
def f(A, x, b):
    assert (np.size(x, 0) == np.size(A, 1) and np.size(A, 0) == np.size(b, 0) and \
            np.size(x, 1) == np.size(b, 1) == 1)
    q = (x - b).T
    w = x - b
    return np.dot(np.dot(q, A), w)[0, 0]


# gradient of f(x)
def grf(A, x, b):
    assert (np.size(x, 0) == np.size(A, 1) and np.size(A, 0) == np.size(b, 0) and \
            np.size(x, 1) == np.size(b, 1) == 1)
    q = 6 * (x - b)[0, 0] + (x - b)[1, 0]
    w = (x-b)[1, 0] * 2 + (x - b)[0, 0]
    e = np.mat('0; 0', dtype=float)
    e[0, 0] = q
    e[1, 0] = w
    return e


def main():
    # Define parameters. Size of A is n x p
    p = 2
    n = 2
    kMax = 500  # Number of iteration

    A = np.random.randn(n, p)
    b = np.random.randn(n, 1)
    epsilon = 0.02
    ramda = 0.89
    A[0, 0] = 250
    A[0, 1] = 15
    A[1, 0] = 15
    A[1, 1] = 4
    b[0, 0] = 1
    b[1, 0] = 2
    rate = 500 * (1 / max(np.linalg.eigvals(2 * A)))
    xk = np.random.rand(p, 1)
    xk[0, 0] = 1
    xk[1, 0] = 1
    G = np.mat('0; 0', dtype=float)
    bestxk = np.random.rand(p, 1)
    bestxk[0, 0] = 1
    bestxk[1, 0] = 2

    for k in range(1, kMax):
        loss = obj(A, xk, b, ramda)
        G = G + grf(A, xk, b)
        GG = rate/np.sqrt(np.square(G) + epsilon)
        x_kplus1 = xk - np.multiply(GG, grf(A, xk, b))
        loss_plus1 = obj(A, x_kplus1, b, ramda)
        Dobj = np.linalg.norm(loss_plus1 - loss)
        plt.semilogy(k, np.linalg.norm(x_kplus1 - bestxk), 'ro')
        print('k:', k, 'xk: ', x_kplus1, ' obj = ', obj(A, x_kplus1, b, ramda), 'Change = ', Dobj)
        if (Dobj< 0.000001):
            break

        # Update xk
        xk = x_kplus1
    plt.xlabel("k")
    plt.ylabel("wk - bestwk")
    plt.grid(True)
    plt.title("AdaGrad")
    plt.show()


if __name__ == "__main__":
    main()
