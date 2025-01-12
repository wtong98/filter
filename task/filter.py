"""ICL time series learning"""

# <codecell>
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group


class KalmanFilterTask:
    def __init__(self, length=8, n_dims=8, n_obs_dims=None, n_tasks=1, max_sval=1, o_mult=1, t_noise=0.05, o_noise=0.05, batch_size=128, seed=None) -> None:
        self.length = length
        self.n_dims = n_dims
        self.n_obs_dims = n_obs_dims
        self.n_tasks = n_tasks
        self.max_sval = max_sval
        self.o_mult = o_mult
        self.t_noise = t_noise
        self.o_noise = o_noise
        self.batch_size = batch_size
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

        if self.n_obs_dims is None:
            self.n_obs_dims = self.n_dims

        self.t_mat = self.rng.standard_normal((self.n_dims, self.n_dims))
        self.t_mat = self.t_mat / np.linalg.norm(self.t_mat, ord=2) * self.max_sval
        self.o_mat = self.rng.standard_normal((self.n_obs_dims, self.n_dims)) / np.sqrt(n_dims) * self.o_mult
        # self.o_mat = self.o_mat / np.linalg.norm(self.o_mat, ord=2) * self.max_sval

        self.rng = np.random.default_rng(None)
    

    def __next__(self):
        # temporary fix
        if not hasattr(self, 'n_obs_dims'):
            self.n_obs_dims = self.n_dims

        zs = np.random.randn(self.batch_size, self.n_dims, 1) / np.sqrt(self.n_dims)

        t_mat = self.t_mat
        o_mat = self.o_mat

        if self.n_tasks is None:
            t_mat = self.rng.standard_normal((self.batch_size, self.n_dims, self.n_dims))
            t_mat = t_mat / np.linalg.norm(t_mat, ord=2, keepdims=True, axis=(-2, -1)) * self.max_sval
            # print(np.linalg.norm(t_mat, ord=2, axis=(-2, -1)))
            o_mat = self.rng.standard_normal((self.batch_size, self.n_obs_dims, self.n_dims)) / np.sqrt(self.n_dims)
            # o_mat = np.repeat(np.eye(self.n_obs_dims)[None], self.batch_size, axis=0)

        xs_all = []
        for _ in range(self.length):
            xs = o_mat @ zs + np.random.randn(self.batch_size, self.n_obs_dims, 1) * np.sqrt(self.o_noise / self.n_obs_dims)
            zs = t_mat @ zs + np.random.randn(self.batch_size, self.n_dims, 1) * np.sqrt(self.t_noise / self.n_dims)
            xs_all.append(xs)
        
        xs_all = np.stack(xs_all, axis=1)[...,0]
        return xs_all

    def __iter__(self):
        return self


# task = KalmanFilterTask(max_sval=1.5, length=16, batch_size=32, n_tasks=None, n_obs_dims=1, o_noise=0.001, t_noise=0.001)
# xs = next(task)

# plt.plot(np.linalg.norm(xs, axis=-1).T, '--o')

# xs.shape

# <codecell>
def pred_kalman(xs, task, return_mat=False):
    I = np.eye(task.n_dims)
    Io = np.eye(task.n_obs_dims)

    A = task.t_mat
    C = task.o_mat
    S_u = I * task.t_noise / task.n_dims
    S_w = Io * task.o_noise / task.n_obs_dims

    ba = np.zeros((task.n_dims, task.batch_size))
    Sa = S_u.copy()

    preds = []
    Ms = []
    Ns = []
    all_true_mse = []

    for i in range(xs.shape[1]):
        y = xs[:,i].T

        L = Sa @ C.T @ np.linalg.pinv(C @ Sa @ C.T + S_w)

        bp = (I - L @ C) @ ba + L @ y
        Sp = (I - L @ C) @ Sa

        ba = A @ bp
        Sa = A @ Sp @ A.T + S_u

        true_mse = np.trace(C @ Sa @ C.T + S_w) / task.n_dims
        all_true_mse.append(true_mse)

        obs = C @ ba
        preds.append(obs.T)

        M = A @ (I - L @ C)
        N = A @ L

        Ms.append(M)
        Ns.append(N)

    preds = np.stack(preds, axis=1)
    Ms = np.stack(Ms, axis=0)
    Ns = np.stack(Ns, axis=0)

    if return_mat:
        kalman_mat = []
        for i in range(xs.shape[1]):
            row = []
            for t in range(i):
                cum_mat = Ns[t]
                for j in range(t+1, i):
                    cum_mat = Ms[j] @ cum_mat
                row.append(C @ cum_mat)
            kalman_mat.append(row)

            while len(row) < xs.shape[1] - 1:
                row.append(np.zeros((C.shape[0], C.shape[0])))
            
        kalman_mat = kalman_mat[1:]
        kalman_mat = np.block(kalman_mat)
        return preds, kalman_mat
    else:
        return preds, all_true_mse


# ns = [1e-5, 1, 1e3, 1e6]
# max_svals = [0.1, 1, 10]

# fig, axs = plt.subplots(3, 4, figsize=(8, 6))
# for ax, (sval, noise) in zip(axs.ravel(), itertools.product(max_svals, ns)):
#     task = KalmanFilterTask(length=16, n_dims=512, n_obs_dims=32, t_noise=noise, o_noise=noise, max_sval=sval)
#     xs = next(task)
#     _, mat = pred_kalman(xs, task, return_mat=True)

#     ax.imshow(np.abs(mat))
#     ax.set_title(f'ns={noise}, sv={sval}')

# fig.tight_layout()
# plt.savefig('../experiment/fig/kalman_sweep.png')
# %%
