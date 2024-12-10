"""ICL time series learning"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group


class KalmanFilterTask:
    def __init__(self, length=8, n_dims=8, t_noise=0.05, o_noise=0.05, batch_size=128) -> None:
        self.length = length
        self.n_dims = n_dims
        self.t_noise = t_noise
        self.o_noise = o_noise
        self.batch_size = batch_size

        self.t_mat = np.random.randn(self.n_dims, self.n_dims) 
        self.t_mat = self.t_mat / np.linalg.norm(self.t_mat, ord=2)
        self.o_mat = np.random.randn(self.n_dims, self.n_dims) / np.sqrt(n_dims)
    

    def __next__(self):
        zs = np.random.randn(self.batch_size, self.n_dims, 1) / np.sqrt(self.n_dims)

        xs_all = []
        for _ in range(self.length):
            xs = self.o_mat @ zs + np.random.randn(self.batch_size, self.n_dims, 1) * np.sqrt(self.o_noise / self.n_dims)
            zs = self.t_mat @ zs + np.random.randn(self.batch_size, self.n_dims, 1) * np.sqrt(self.t_noise / self.n_dims)
            xs_all.append(xs)
        
        xs_all = np.stack(xs_all, axis=1).squeeze()
        return xs_all

    def __iter__(self):
        return self

# task = KalmanFilterTask(length=30, n_dims=100, batch_size=10)
# xs = next(task)

# xs = np.linalg.norm(xs, axis=-1)
# plt.plot(xs.T, '--o')

# <codecell>
# tasks = np.random.randn(5, 8, 8)
# norms = np.linalg.norm(tasks, ord=2, axis=(-2, -1), keepdims=True)
# norms.shape

# tasks = tasks / norms
# np.linalg.norm(tasks, ord=2, axis=(-2, -1))

# <codecell>
# n_dims = 100
# m = special_ortho_group.rvs(n_dims, size=3)

# flux = 1 + np.random.randn(n_dims) / np.sqrt(n_dims)

# m = m @ np.diag(flux)

# # m = np.random.randn(n_dims, n_dims) / np.sqrt(n_dims)

# # m = m @ m.T
# sv = np.linalg.norm(m, ord=2)

# # m = m / sv
# sv


# x = np.random.randn(n_dims, 1) / np.sqrt(n_dims)
# x2 = x + np.random.randn(n_dims, 1) / np.sqrt(n_dims) / 2

# vals = []
# vals2 = []
# for _ in range(100):
#     vals.append(np.linalg.norm(x))
#     vals2.append(np.linalg.norm(x2))
#     x = m @ x
#     x2 = m @ x2

# xs = np.arange(len(vals))

# plt.plot(vals, '--o')
# plt.plot(vals2, '--o')
# plt.yscale('log')


# %%
