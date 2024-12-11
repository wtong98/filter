"""ICL time series learning"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group


class KalmanFilterTask:
    def __init__(self, length=8, n_dims=8, n_tasks=None, t_noise=0.05, o_noise=0.05, batch_size=128, seed=None) -> None:
        self.length = length
        self.n_dims = n_dims
        self.n_tasks = n_tasks
        self.t_noise = t_noise
        self.o_noise = o_noise
        self.batch_size = batch_size
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

        self.t_mat = self.rng.standard_normal((self.n_dims, self.n_dims))
        self.t_mat = self.t_mat / np.linalg.norm(self.t_mat, ord=2)
        self.o_mat = self.rng.standard_normal((self.n_dims, self.n_dims)) / np.sqrt(n_dims)

        self.rng = np.random.default_rng(None)
    

    def __next__(self):
        zs = np.random.randn(self.batch_size, self.n_dims, 1) / np.sqrt(self.n_dims)

        t_mat = self.t_mat
        o_mat = self.o_mat

        if self.n_tasks is None:
            t_mat = self.rng.standard_normal((self.batch_size, self.n_dims, self.n_dims))
            t_mat = self.t_mat / np.linalg.norm(self.t_mat, ord=2, keepdims=True)
            o_mat = self.rng.standard_normal((self.batch_size, self.n_dims, self.n_dims)) / np.sqrt(self.n_dims)

        xs_all = []
        for _ in range(self.length):
            xs = o_mat @ zs + np.random.randn(self.batch_size, self.n_dims, 1) * np.sqrt(self.o_noise / self.n_dims)
            zs = t_mat @ zs + np.random.randn(self.batch_size, self.n_dims, 1) * np.sqrt(self.t_noise / self.n_dims)
            xs_all.append(xs)
        
        xs_all = np.stack(xs_all, axis=1).squeeze()
        return xs_all

    def __iter__(self):
        return self


# task = KalmanFilterTask(batch_size=5, n_tasks=None)
# xs = next(task)

# plt.plot(np.linalg.norm(xs, axis=-1).T, '--o')

# <codecell>


def pred_kalman(xs, task):
    I = np.eye(task.n_dims)
    A = task.t_mat
    C = task.o_mat
    S_u = I * task.t_noise / task.n_dims
    S_w = I * task.o_noise / task.n_dims

    ba = np.zeros((task.n_dims, task.batch_size))
    Sa = S_u.copy()

    preds = []

    # begin loop
    for i in range(xs.shape[1]):
        y = xs[:,i].T

        L = Sa @ C.T @ np.linalg.pinv(C @ Sa @ C.T + S_w)
        bp = (I - L @ C) @ ba + L @ y
        Sp = (I - L @ C) @ Sa

        ba = A @ bp
        Sa = A @ Sp @ A.T + S_u

        obs = C @ ba
        preds.append(obs.T)

    preds = np.stack(preds, axis=1)
    return preds

# task = KalmanFilterTask(t_noise=0.25, o_noise=0.25)
# xs = next(task)
# preds = pred_kalman(xs, task)

# xs = xs[:,1:]
# preds = preds[:,:-1]

# k_mse = ((xs - preds)**2).mean(axis=(0, -1))
# z_mse = ((xs)**2).mean(axis=(0, -1))

# print(k_mse)
# print(z_mse)


