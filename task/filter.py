"""ICL time series learning"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group


class KalmanFilterTask:
    def __init__(self, length=8, n_dims=8, matrix_dist='orthogonal', batch_size=128) -> None:
        self.length = length
        self.n_dims = n_dims
        self.matrix_dist = matrix_dist
        self.batch_size = batch_size
    
    def __next__(self):
        if self.matrix_dist == 'orthogonal':
            ms = special_ortho_group.rvs(self.n_dims, size=self.batch_size)
        else:
            ms = np.random.randn(self.batch_size, self.n_dims, self.n_dims) / np.sqrt(self.n_dims)

        flux = np.stack([np.diag(1 + np.random.randn(self.n_dims) / np.sqrt(self.n_dims)) for _ in range(self.batch_size)], axis=0)
        ms = ms @ flux

        xs = np.random.randn(self.batch_size, self.n_dims, 1) / np.sqrt(self.n_dims)

        xs_all = []
        for _ in range(self.length):
            xs_all.append(xs)
            xs = ms @ xs
        
        xs_all = np.stack(xs_all, axis=1).squeeze()
        return xs_all

    def __iter__(self):
        return self

# task = KalmanFilterTask(length=30, n_dims=100, batch_size=10)
# xs = next(task)
# xs

# xs = np.linalg.norm(xs, axis=-1)
# plt.plot(xs.T, '--o')


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
