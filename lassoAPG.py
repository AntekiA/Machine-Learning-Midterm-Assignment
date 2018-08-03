import numpy as np
import matplotlib.pyplot as plt


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


# Model function evaluated at x and touches f(x) in xk
def m(x, xk, A, b, GammaK):
    assert (np.size(xk, 0) == np.size(x, 0) == np.size(A, 1) \
            and np.size(A, 0) == np.size(b, 0) and \
            np.size(xk, 1) == np.size(x, 1) == np.size(b, 1) == 1 and np.isscalar(GammaK))
    innerProd = grf(A, xk, b).T.dot(x - xk)
    xDiff = x - xk
    return f(A, xk, b) + innerProd + ((1.0 / 2.0) * GammaK) * xDiff.T.dot(xDiff)


# Shrinkage or Proximal operation
def proxNorm1(y, ramda):
    assert (np.size(y, 1) == 1)
    return np.multiply(np.sign(y), np.maximum(np.zeros(np.shape(y)), np.abs(y) - ramda))


def main():
    # Define parameters. Size of A is n x p
    p = 2
    n = 2
    kMax = 500  # Number of iteration

    A = np.random.randn(n, p)
    b = np.random.randn(n, 1)
    ramda = 6
    A[0, 0] = 3
    A[0, 1] = 0.5
    A[1, 0] = 0.5
    A[1, 1] = 1
    b[0, 0] = 1
    b[1, 0] = 2
    # Proximal Gradient Descent
    xk = np.random.rand(p, 1)   # Initialize with random
    xk[0, 0] = 3
    xk[1, 0] = -1

    xk_acc = xk.copy()
    yk_acc = xk_acc
    tk_acc = 1
    bestxk = np.random.rand(p, 1)
    bestxk[0, 0] = 0.33
    bestxk[1, 0] = 0

    for k in range(1, kMax):

        # =================== Accelerated Proximal GD ========================
        Gammak = 1 / max(np.linalg.eigvals(2 * A))

        x_kplus1_acc = yk_acc - Gammak * grf(A, yk_acc, b)
        x_kplus1_acc = proxNorm1(x_kplus1_acc, Gammak * ramda)
        t_kplus1_acc = 0.5 + 0.5 * np.sqrt(1 + 4 * (tk_acc ** 2.))
        y_kplus1_acc = x_kplus1_acc + ((tk_acc - 1) / (tk_acc + 1)) * \
                       (x_kplus1_acc - xk_acc)

        # Change in the value of objective function for this iteration
        Dobj_acc = np.linalg.norm(obj(A, x_kplus1_acc, b, ramda) - \
                                  obj(A, xk_acc, b, ramda))
        plt.semilogy(k, np.linalg.norm(x_kplus1_acc - bestxk), 'ro')
        print('k:', k,  'xk: ', x_kplus1_acc, ' obj = ', obj(A, x_kplus1_acc, b, ramda), 'Change = ', Dobj_acc)
        if (Dobj_acc < 0.000001):
            break

        # Update
        xk_acc = x_kplus1_acc
        tk_acc = t_kplus1_acc
        yk_acc = y_kplus1_acc
        # ====================================================================
    plt.xlabel("k")
    plt.ylabel("wk - bestwk")
    plt.grid(True)
    plt.title("ramuda = 6")
    plt.show()


if __name__ == "__main__":
    main()
