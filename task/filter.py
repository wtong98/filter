"""ICL time series learning"""

# <codecell>
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group

import sys
sys.path.append('../')
from common import t as tp


class KalmanFilterTask:
    def __init__(self, length=8, n_dims=8, n_obs_dims=None, 
                 mode=None, n_tasks=1, n_snaps=None,
                 max_sval=1, o_mult=1, t_noise=0.05, o_noise=0.05, 
                 batch_size=128, seed=None) -> None:
        
        self.length = length
        self.n_dims = n_dims
        self.n_obs_dims = n_obs_dims
        self.mode = mode
        self.n_tasks = n_tasks
        self.n_snaps = n_snaps
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
        zs_init = np.copy(zs)

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

        if self.mode == 'cheat':
            return xs_all, zs_init
        elif self.mode == 'ac':
            assert len(t_mat.shape) == 3

            if self.n_snaps is not None:
                zs_all = []
                zs = np.random.randn(self.batch_size, self.n_dims, 1) / np.sqrt(self.n_dims)
                for _ in range(self.n_snaps):
                    zs = t_mat @ zs + np.random.randn(self.batch_size, self.n_dims, 1) * np.sqrt(self.t_noise / self.n_dims)
                    zs_all.append(zs)
                
                zs_all = np.stack(zs_all, axis=1).squeeze()
                V_pre = zs_all[:,:-1]
                V_post = zs_all[:,1:]

                V_pre_inv = np.linalg.pinv(V_pre)
                t_mat_est = tp(V_pre_inv @ V_post)
                # print(np.round(t_mat[0], decimals=2))
                t_mat = t_mat_est

            xs_full = np.zeros((self.batch_size, self.n_dims + self.n_obs_dims + self.length, self.n_dims + self.n_obs_dims))
            xs_full[:, :self.n_dims, self.n_obs_dims:] = t_mat
            xs_full[:, self.n_dims:(self.n_dims+self.n_obs_dims), self.n_obs_dims:] = o_mat
            xs_full[:, -self.length:, :self.n_obs_dims] = xs_all

            xs_all = xs_full

        return xs_all

    def __iter__(self):
        return self

# batch_size = 3
# n_dims = 4
# t_noise = 0.00001
# n_snaps = 10

# t_mat = np.random.randn(batch_size, n_dims, n_dims)
# t_mat = t_mat / np.linalg.norm(t_mat, ord=2, keepdims=True, axis=(-2, -1))

# zs_all = []
# zs = np.random.randn(batch_size, n_dims, 1) / np.sqrt(n_dims)
# for _ in range(n_snaps + 1):
#     zs = t_mat @ zs + np.random.randn(batch_size, n_dims, 1) * np.sqrt(t_noise / n_dims)
#     zs_all.append(zs)

# zs_all = np.stack(zs_all, axis=1).squeeze()

# V_pre = zs_all[:,:-1]
# V_post = zs_all[:,1:]

# V_pre_inv = np.linalg.pinv(V_pre)
# t_mat_est = tp(V_pre_inv @ V_post)


# task = KalmanFilterTask(
#     mode='ac',
#     n_snaps=32,
#     max_sval=1.5, 
#     length=16, 
#     batch_size=32, 
#     n_tasks=None, 
#     n_dims=4, n_obs_dims=1,
#     o_noise=0.05, 
#     t_noise=0.05)

# xs = next(task)

# # plt.plot(np.linalg.norm(xs, axis=-1).T, '--o')

# np.round(xs[0], decimals=2)

# mask = (xs[...,0] != 0)
# np.expand_dims(mask, axis=-1).shape

# <codecell>

def mat_vec_prod(M, v):
    if len(M.shape) == 2:
        return M @ v
    else:
        return np.einsum('bij,jb->ib', M, v)


def pred_kalman(xs, task, A=None, C=None, return_mat=False):
    I = np.eye(task.n_dims)
    Io = np.eye(task.n_obs_dims)

    if A is None or C is None:
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

        L = Sa @ tp(C) @ np.linalg.pinv(C @ Sa @ tp(C) + S_w)

        bp = mat_vec_prod((I - L @ C), ba) + mat_vec_prod(L, y)
        Sp = (I - L @ C) @ Sa

        ba = mat_vec_prod(A, bp)
        Sa = A @ Sp @ tp(A) + S_u

        true_mse = np.trace(C @ Sa @ tp(C) + S_w, axis1=-2, axis2=-1).mean() / task.n_dims

        all_true_mse.append(true_mse)

        obs = mat_vec_prod(C, ba)
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


# task = KalmanFilterTask(
#     # n_tasks=1,
#     n_tasks=None,
#     mode='ac',
#     n_obs_dims=8, 
#     max_sval=1,
#     n_dims=8,
#     length=32,
#     t_noise=0.05,
#     o_noise=0.05,
#     batch_size=512)

# xs = next(task)
# t_mats = xs[:,:8,8:]
# o_mats = xs[:,8:16,8:]
# xs = xs[:,16:,:8]

# xs_pred, true_mse = pred_kalman(xs, task, A=t_mats, C=o_mats)
# # xs_pred, true_mse = pred_kalman(xs, task)

# mse = ((xs[:,1:] - xs_pred[:,:-1])**2).mean(axis=(0, -1))
# zer = ((xs[:, 1:])**2).mean(axis=(0, -1))

# idx = 2

# plt.plot(mse[idx:], '--o')
# plt.plot(true_mse[idx:], '--o')
# plt.plot(zer[idx:], '--o')

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
