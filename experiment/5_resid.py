"""Modeling non-Markovian sequences with multilayer Transformers + residual connections"""

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
from model.transformer import TransformerConfig, sinusoidal_init
from task.filter import KalmanFilterTask, pred_kalman

sin_init = sinusoidal_init()

set_theme()

# <codecell>
length = 16
n_dims = 3
n_obs_dims = 1
n_hidden = 256
n_heads = 4
n_layers = 2

noise = 0.001

# seed = new_seed()
seed = 3930462014
train_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)
test_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)


config = TransformerConfig(n_layers=n_layers,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=0,
                           n_heads=1,
                           layer_norm=False,
                           residual_connections=True,
                           freeze_emb=False,
                           return_final_logits_only=False,
                           n_out=n_obs_dims)

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_every=1000,
                    train_iters=5_000, 
                    seed=None)

# <codecell>
train_task.batch_size = 4096
train_task.length = 64

xs = next(train_task)

pred = state.apply_fn({'params': state.params}, xs)

pred_naive = xs[:,:-1]
pred = pred[:,:-1]

xs_k = xs
pred_k, kalman_mat = pred_kalman(xs_k, train_task, return_mat=True)
xs_k = xs_k[:,1:]
pred_k = pred_k[:,:-1]

xs = xs[:,1:]

pred_mse = ((xs - pred)**2).mean(axis=(0, -1))
zero_mse = (xs**2).mean(axis=(0, -1))
kalman_mse = ((xs_k - pred_k)**2).mean(axis=(0, -1))

plt.plot(pred_mse[:], '--o', label='Transformer', alpha=0.7, color='C0')
plt.plot(kalman_mse[:], '--o', label='Kalman', alpha=0.7, color='C1')
plt.plot(zero_mse[:], '--o', label='Zero', alpha=0.7, color='C8')

plt.axvline(x=14, linestyle='dashed', color='gray')

plt.legend(bbox_to_anchor=(1,1))
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.tight_layout()

plt.savefig('fig/extrapolation.png')

# %%
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

V1 = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
O1 = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()
M1 = V1 @ O1

V2 = state.params['TransformerBlock_1']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
O2 = state.params['TransformerBlock_1']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()
M2 = V2 @ O2

Z = np.kron(atts[1] @ atts[0], (E @ M1 @ M2 @ D).T) \
    + np.kron(atts[1], (E @ M2 @ D).T) \
    + np.kron(atts[0], (E @ M1 @ D).T) \
    + np.kron(np.eye(atts[0].shape[1]), (E @ D).T) \

ps = Z @ xs.reshape(xs.shape[0], -1, 1)

print(pred[0,0])
print(ps[0,0])

np.mean((pred - ps)**2)

# <codecell>
plt.imshow(Z[3])
plt.colorbar()
plt.savefig('fig/transformer_coeff.png')

# <codecell>
kalman_mat = kalman_mat[:16, :16]
plt.imshow(kalman_mat)
plt.colorbar()
plt.savefig('fig/kalman_coeff.png')
# %%
all_ms = []
for big_mat in Z:
    big_mat_vals = big_mat[~np.isclose(kalman_mat, 0)]
    kal_mat_vals = kalman_mat[~np.isclose(kalman_mat, 0)]

    m = np.linalg.lstsq(big_mat_vals[:,None], kal_mat_vals)[0]
    all_ms.append(m)

big_mat_mean = np.mean(Z, axis=0)
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

# for m_sel in all_ms:
#     plt.plot(vals, m_sel * vals, color='lightpink', alpha=0.05)

m = np.mean(all_ms)

# plt.plot(vals, m * vals, color='red', alpha=0.5)
plt.scatter(big_mat_vals, kal_mat_vals, alpha=0.7)

# plt.text(-0.17, 0.1, f'$r^2 = {r2:.2f}$', color='red', fontsize=10)
plt.xlabel('Transformer coefficients')
plt.ylabel('Kalman coefficients')

plt.tight_layout()
plt.savefig('fig/transformer_v_kalman_scatter.png')
# %%
