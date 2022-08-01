import matplotlib.pyplot as plt
import numpy as np

# Currently only batch steepest gradient is implemented

def get_posterior(y, w, x):
    exp_logit = np.exp(np.matmul(x, w))
    return exp_logit[np.arange(len(x)), y] / exp_logit.sum(axis=1)

def get_grad(y, x, p, lam, w):
    exp_logit = np.exp(np.matmul(x, w))
    prob = -exp_logit / exp_logit.sum(axis=1, keepdims=True)
    prob[np.arange(len(x)), y] += 1
    return -np.einsum("nd, nc->ndc", x, prob).mean(axis=0) + lam * w

def get_loss(p, lam, w):
    return np.mean(-np.log(p)) + lam * np.sum(w * w) / 2



# Generate dataset
np.random.seed(0)
n = 200
x = 3 * (np.random.rand(n, 4) - 0.5)
W = np.array([[ 2,  -1, 0.5,],
              [-3,   2,   1,],
              [ 1,   2,   3]],)
y = np.argmax(np.dot(np.hstack([x[:,:2], np.ones((n, 1))]), W.T)
                        + 0.5 * np.random.randn(n, 3), axis=1)

# Hyperparameter
num_iter = 4000
alpha = 0.5
lam = 0.01

# Data process
w_grad = np.random.randn(5, 3)
w_newton = w_grad.copy()
x1 = np.ones((n, 5))
x1[:, 0:4] = x
x = x1

# Batch steepest gradient
loss_hist_batch = []
for _ in range(num_iter):
    posterior = get_posterior(y, w_grad, x)
    grad = get_grad(y, x, posterior, lam, w_grad)
    direction = -grad
    loss = get_loss(posterior, lam, w_grad)
    loss_hist_batch.append(loss)
    w_grad += alpha * direction

# # Newton
# loss_hist_newton = []
# for _ in range(num_iter):
#     posterior = get_posterior(y, w_newton, x)
#     grad = get_grad(y, x, posterior, lam, w_newton)
#     hess = get_hessian(posterior, x, lam, 5)
#     direction = np.matmul(np.linalg.inv(hess), -grad)
#     loss = get_loss(posterior, lam, w_newton)
#     loss_hist_newton.append(loss)
#     w_newton += alpha * direction

print(w_grad)
# print(w_newton)

eps = 1e-15
plt.semilogy(np.array(loss_hist_batch) - loss + eps)
# plt.semilogy(np.array(loss_hist_newton) - loss + eps)
plt.xlabel("t")
plt.ylabel("J(wt) - J(w)")
plt.title("Speed of optimization methods")
plt.legend(["Batch steepest gradient"])
plt.show()
