"""Inspecting filter coefficients"""

# <codecell>
from pathlib import Path

from flax import traverse_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from common import *
from train import *
from model.transformer import TransformerConfig 
from task.filter import KalmanFilterTask, pred_kalman

set_theme()

# <codecell>
length = 16
n_dims = 32
n_obs_dims = 32
n_layers = 2

noise = 0.1

seed = new_seed()
train_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)
test_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)


config = TransformerConfig(n_layers=n_layers,
                           n_hidden=512,
                           pos_emb=False,
                           n_mlp_layers=0,
                           n_heads=1,
                           layer_norm=False,
                           residual_connections=True,
                           freeze_emb=False,
                           return_final_logits_only=False,
                        #    use_simple_att=True,
                           n_out=n_obs_dims)


# xs = next(train_task)
# model = config.to_model()
# params = model.init(jax.random.PRNGKey(3), xs)['params']

# out = model.apply({'params': params}, xs)
# out.shape

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_every=1000,
                    train_iters=5_000, 
                    seed=None)

# <codecell>
train_task.n_tasks = 1
train_task.batch_size = 128
train_task.length = 64

xs = next(train_task)
pred = state.apply_fn({'params': state.params}, xs)


# xs = np.linalg.norm(xs, axis=-1)
# pred = np.linalg.norm(pred, axis=-1)

pred_naive = xs[:,:-1]
pred = pred[:,:-1]

train_task.n_tasks = 1
# xs_k = next(train_task)
xs_k = xs
pred_k, _ = pred_kalman(xs_k, train_task)
xs_k = xs_k[:,1:]
pred_k = pred_k[:,:-1]

xs = xs[:,1:]

pred_mse = ((xs - pred)**2).mean(axis=(0, -1))
naive_mse = ((xs - pred_naive)**2).mean(axis=(0, -1))
zero_mse = (xs**2).mean(axis=(0, -1))
kalman_mse = ((xs_k - pred_k)**2).mean(axis=(0, -1))

# normalize by magnitude
# pred_mse /= zero_mse
# kalman_mse /= zero_mse
# zero_mse /= zero_mse

plt.plot(pred_mse[1:], '--o', label='pred', alpha=0.7)
# plt.plot(naive_mse, '--o', label='naive', alpha=0.7)
plt.plot(zero_mse[1:], '--o', label='zero', alpha=0.7)
plt.plot(kalman_mse[1:], '--o', label='kalman', alpha=0.7)

plt.axvline(x=14, linestyle='dashed', color='gray')

plt.legend()
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('mse')
plt.tight_layout()

# plt.savefig('fig/extrapolation_4_layer.png')

# <codecell>
train_task.n_tasks = 1
train_task.batch_size = 1024
train_task.length = 16

xs = next(train_task)
pred, info = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

atts = [info['intermediates'][f'TransformerBlock_{i}']['MultiHeadDotProductAttention_0']['attention_weights'][0].squeeze() for i in range(n_layers)]

for i, att in enumerate(atts):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    for ax, a in zip(axs.ravel(), att):
        im = ax.imshow(a)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    # plt.savefig(f'fig/attention_{i}.png')
    # plt.clf()


# <codecell>
E = state.params['Dense_0']['kernel']
D = state.params['Dense_1']['kernel']
M = E

for i in range(n_layers):
    V = state.params[f'TransformerBlock_{i}']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
    O = state.params[f'TransformerBlock_{i}']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()
    M = M @ V @ O
M = M @ D

# Ms = []
# for i in range(n_layers):
#     M_c = np.array(atts[i][:,:,None,:,None] * M.T[None, None, :, None, :])
#     big_d = train_task.length * train_task.n_dims
#     M_c = M_c.reshape((train_task.batch_size, big_d, big_d))
#     Ms.append(M_c)

# # <codecell>
# plt.imshow(np.abs(Ms[0][9]))
# # plt.imshow(atts[0][9])

# # <codecell>
# xs_long = xs.reshape(xs.shape[0], -1, 1)
# ps_long = Ms[1] @ Ms[0] @ xs_long
# ps = ps_long.reshape(ps_long.shape[0], train_task.length, train_task.n_dims)

full_att = atts[1] @ atts[0]
ps = full_att @ xs @ M

print(pred[0,0])
print(ps[0,0])

np.mean((pred - ps)**2)


# <codecell>
idx = 0

dists = []

big_d = (xs.shape[1] - 1) * xs.shape[2]
big_mats_acc = np.zeros((big_d, big_d))
all_ms = []

for idx in tqdm(range(xs.shape[0])):
    x = xs[idx][:-1]
    A = full_att[idx]

    big_mat = []
    for i in range(A.shape[0] - 1):
        row = []
        for j in range(A.shape[1] - 1):
            fac = A[i,j]
            # fac = A[i,j] if i == j else 0
            # fac = 1 if i == j else 0
            row.append(fac * M.T)
        
        big_mat.append(row)

    big_mat = np.block(big_mat)
    big_mats_acc += big_mat

    x = x.reshape(-1, 1)
    x_pred = big_mat @ x
    x_pred = x_pred.reshape(A.shape[0], -1)

    k_preds, kalman_mat = pred_kalman(xs, train_task, return_mat=True)

    big_mat_vals = big_mat[~np.isclose(kalman_mat, 0)]
    kal_mat_vals = kalman_mat[~np.isclose(kalman_mat, 0)]

    m = np.linalg.lstsq(big_mat_vals[:,None], kal_mat_vals)[0]
    all_ms.append(m)

    # k_preds = k_preds[idx]

    # x = x.reshape(-1, 1)
    # x_pred = kalman_mat @ x[:480]
    # x_pred = x_pred.reshape(A.shape[0], -1)

    # kalman_mag = np.mean(kalman_mat**2) * 2
    # dists.append(2 * np.mean(np.sqrt((big_mat - kalman_mat)**2 / kalman_mag)))
    dists.append(np.mean((big_mat - kalman_mat)**2))

# <codecell>
big_mat_mean = big_mats_acc / xs.shape[0]

big_mat_vals = big_mat_mean[~np.isclose(kalman_mat, 0)]
kal_mat_vals = kalman_mat[~np.isclose(kalman_mat, 0)]

# m = np.linalg.lstsq(big_mat_vals[:,None], kal_mat_vals)[0]
r2 = np.corrcoef(big_mat_vals, kal_mat_vals)[0,1]**2

vals = np.linspace(-0.2, 0.2, 500)

all_ms = np.sort(all_ms)
low_idx = int(0.025 * len(all_ms))
hi_idx = int(0.975 * len(all_ms))
all_ms = all_ms[low_idx:hi_idx]

np.random.shuffle(all_ms)

for m_sel in all_ms:
    plt.plot(vals, m_sel * vals, color='lightpink', alpha=0.05)

m = np.mean(all_ms)

plt.plot(vals, m * vals, color='red', alpha=0.5)
plt.scatter(big_mat_vals, kal_mat_vals, alpha=0.03, s=1)

plt.text(-0.17, 0.1, f'$r^2 = {r2:.2f}$', color='red', fontsize=10)

plt.xlabel('Transformer coefficients')
plt.ylabel('Kalman coefficients')

plt.tight_layout()
# plt.savefig('fig/transformer_vs_kalman_reg_coeff.png')
# plt.savefig('fig/transformer_vs_kalman_reg_coeff.svg')
# plt.savefig('fig/transformer_vs_kalman_reg_coeff.pdf')

# <codecell>
### REDONE BUT AS A MEGA PLOT
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

for run_id in range(3):
    dists = []
    big_d = (xs.shape[1] - 1) * xs.shape[2]
    big_mats_acc = np.zeros((big_d, big_d))
    all_ms = []

    for idx in tqdm(range(xs.shape[0])):
        x = xs[idx][:-1]
        A = full_att[idx]

        big_mat = []
        for i in range(A.shape[0] - 1):
            row = []
            for j in range(A.shape[1] - 1):
                if run_id == 0:
                    fac = A[i,j]
                elif run_id == 1:
                    fac = A[i,j] if i == j else 0
                elif run_id == 2:
                    fac = 1 if i == j else 0
                row.append(fac * M.T)
            
            big_mat.append(row)

        big_mat = np.block(big_mat)
        big_mats_acc += big_mat

        x = x.reshape(-1, 1)
        x_pred = big_mat @ x
        x_pred = x_pred.reshape(A.shape[0], -1)

        k_preds, kalman_mat = pred_kalman(xs, train_task, return_mat=True)

        big_mat_vals = big_mat[~np.isclose(kalman_mat, 0)]
        kal_mat_vals = kalman_mat[~np.isclose(kalman_mat, 0)]

        m = np.linalg.lstsq(big_mat_vals[:,None], kal_mat_vals)[0]
        all_ms.append(m)

        dists.append(np.mean((big_mat - kalman_mat)**2))


    big_mat_mean = big_mats_acc / xs.shape[0]

    big_mat_vals = big_mat_mean[~np.isclose(kalman_mat, 0)]
    kal_mat_vals = kalman_mat[~np.isclose(kalman_mat, 0)]

    r2 = np.corrcoef(big_mat_vals, kal_mat_vals)[0,1]**2

    vals = np.linspace(-0.2, 0.2, 500)

    all_ms = np.sort(all_ms)
    low_idx = int(0.025 * len(all_ms))
    hi_idx = int(0.975 * len(all_ms))
    all_ms = all_ms[low_idx:hi_idx]

    np.random.shuffle(all_ms)

    for m_sel in all_ms:
        axs[run_id].plot(vals, m_sel * vals, color='lightpink', alpha=0.05)

    m = np.mean(all_ms)

    axs[run_id].plot(vals, m * vals, color='red', alpha=0.5)
    axs[run_id].scatter(big_mat_vals, kal_mat_vals, alpha=0.03, s=1)

    axs[run_id].text(-0.17, 0.1, f'$r^2 = {r2:.2f}$', color='red', fontsize=10)

    axs[run_id].set_xlabel('Transformer coefficients')
    axs[run_id].set_ylabel('Kalman coefficients')

    if run_id == 0:
        name = 'Original att'
    elif run_id == 1:
        name = 'Diagonal only'
    elif run_id == 2:
        name = 'Identity att'
    
    axs[run_id].set_title(name)

fig.tight_layout()
plt.savefig('fig/transformer_v_kalman_vary_att_2_layer.png')

# <codecell>
plt.title('Abs Kalman Matrix')
plt.imshow(np.abs(kalman_mat))
plt.colorbar()
plt.savefig('fig/kalman_mat.png')

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

    all_Ls = []

    for i in range(xs.shape[1]):
        y = xs[:,i].T

        L = Sa @ C.T @ np.linalg.pinv(C @ Sa @ C.T + S_w)
        all_Ls.append(L)

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
    
    return all_Ls

Ls = pred_kalman(xs, train_task)

all_diffs = []
for i in range(len(Ls) - 1):
    sq_dff = np.mean(np.abs(Ls[i+1] - Ls[i]))
    all_diffs.append(sq_dff)

plt.plot(all_diffs, '--o')
plt.title('Coordinate-wise L abs diff')
plt.xlabel('Time')
plt.ylabel('|Diff|')
plt.savefig('fig/kalman_gain_diff.png')