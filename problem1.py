import matplotlib.pyplot as plt
import numpy as np

def get_posterior(y, w, x):
    return 1 / (1 + np.exp(-y * np.matmul(x, w)))

def get_grad(y, x, p, lam, w):
    return np.mean(-y * x * (1 - p), axis=0, keepdims=True).T + lam * w

def get_loss(p, lam, w):
    return np.mean(np.log(1 / p)) + 0.5 * lam * np.sum(w * w)

def get_hessian(p, x, lam, d):
    p1_p = posterior * (1 - posterior)
    xxT = np.einsum("Bi, Bj->Bij", x, x)
    lamI = lam * np.eye(d)
    hessian = np.mean(p1_p[..., np.newaxis] * xxT, axis=0, keepdims=True) + lamI
    return hessian.squeeze()


# Generate dataset
np.random.seed(0)
n = 200
x = 3 * (np.random.rand(n, 4) - 0.5)
y = (2 * x[:, 0] - 1 * x[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y = 2 * y - 1

# Hyperparameter
num_iter = 1000
alpha = 0.5
lam = 0.01

# Data process
w_grad = np.random.randn(5, 1)
w_newton = w_grad.copy()
x1 = np.ones((n, 5))
x1[:, 0:4] = x
x = x1
y = y.reshape((n, 1))  # For broadcasting

# Batch steepest gradient
loss_hist_batch = []
for _ in range(num_iter):
    posterior = get_posterior(y, w_grad, x)
    grad = get_grad(y, x, posterior, lam, w_grad)
    direction = -grad
    loss = get_loss(posterior, lam, w_grad)
    loss_hist_batch.append(loss)
    w_grad += alpha * direction

# Newton
loss_hist_newton = []
for _ in range(num_iter):
    posterior = get_posterior(y, w_newton, x)
    grad = get_grad(y, x, posterior, lam, w_newton)
    hess = get_hessian(posterior, x, lam, 5)
    direction = np.matmul(np.linalg.inv(hess), -grad)
    loss = get_loss(posterior, lam, w_newton)
    loss_hist_newton.append(loss)
    w_newton += alpha * direction

print(w_grad)
print(w_newton)

eps = 1e-16
plt.semilogy(np.array(loss_hist_batch) - loss + eps)
plt.semilogy(np.array(loss_hist_newton) - loss + eps)
plt.xlabel("t")
plt.ylabel("J(wt) - J(w)")
plt.title("Speed of optimization methods")
plt.legend(["Batch steepest gradient", "Newton"])
plt.show()
