import matplotlib.pyplot as plt
import numpy as np


def soft_thresh(w, q):
    res = w.copy()
    for i, wi in enumerate(w):
        if wi > q:
            res[i] = wi - q
        elif abs(wi) <= q:
            res[i] = 0
        else:
            res[i] = wi + q
    return res


def get_curve(lam):
    A = np.array([[3, 0.5], [0.5, 1]])
    mu = np.array([[1], [2]])

    x_init = np.array([[3], [-1]])
    lr = 1 / np.max(np.linalg.eig(2 * A)[0])

    x_history = [x_init.T]
    xt = x_init
    for _ in range(150):
        grad = 2 * np.dot(A, xt - mu)
        xt = soft_thresh(xt - lr * grad, lam * lr)
        x_history.append(xt.T)

    x_history = np.vstack(x_history)

    eps = 1e-16
    return abs(x_history - x_history[-1]).sum(axis=1) + eps

plt.semilogy(get_curve(2))
plt.semilogy(get_curve(4))
plt.semilogy(get_curve(6))
plt.xlabel("t")
plt.ylabel("sum(abs(wt-w))")
plt.title("Proximal gradient for LASSO")
plt.legend(["λ=2", "λ=4", "λ=6"])
plt.show()
