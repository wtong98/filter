"""Experimenting with providing AC matrices"""

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

# <codecell>
length = 16
n_dims = 4
n_obs_dims = 4
n_hidden = 256
n_heads = 1
n_layers = 2

noise = 0.1

seed = new_seed()

train_task = KalmanFilterTask(length=length, 
                              mode='ac',
                              n_snaps=None,
                              n_obs_dims=n_obs_dims, 
                              n_tasks=None, 
                              n_dims=n_dims, 
                              t_noise=1, 
                              o_noise=0.01, 
                              seed=seed, 
                              noise_dist='gauss',
                              nonlin='tanh',
                              max_sval=1)


config = TransformerConfig(n_layers=n_layers,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=n_heads,
                           layer_norm=True,
                           residual_connections=True,
                           freeze_emb=False,
                           return_final_logits_only=False,
                           n_out=n_obs_dims)

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_every=1000,
                    train_iters=10_000, 
                    seed=None)


# <codecell>
train_task.batch_size = 512
train_task.length = 32

xs = next(train_task)
A = xs[:,:n_dims,n_obs_dims:]
C = xs[:,n_dims:(n_dims+n_obs_dims),n_obs_dims:]
xs_vals = xs[:,(n_dims + n_obs_dims):, :n_obs_dims]

pred = state.apply_fn({'params': state.params}, xs)
pred = pred[:,:-1]

xs_k = xs_vals
pred_k, true_mse = pred_kalman(xs_k, train_task, A=A, C=C, t_noise=1)
xs_k = xs_k[:,1:]
pred_k = pred_k[:,:-1]

xs = xs[:,1:,:pred.shape[-1]]
xs_vals = xs_vals[:,1:]

pred_mse = ((xs - pred)**2).mean(axis=(0, -1))
pred_mse = pred_mse[(n_dims + n_obs_dims):]

zero_mse = (xs_vals**2).mean(axis=(0, -1))
kalman_mse = ((xs_k - pred_k)**2).mean(axis=(0, -1))

start_idx = 1
true_mse = np.array(true_mse)
plt.plot(pred_mse[start_idx:], '--o', label='Transformer', alpha=0.7, color='C0')
plt.plot(kalman_mse[start_idx:], '--o', label='Kalman', alpha=0.7, color='C1')
plt.plot(zero_mse[start_idx:], '--o', label='Zero', alpha=0.7, color='C8')
plt.plot(true_mse[start_idx:], '--', label='Kalman Var', alpha=0.7, color='red')

plt.axvline(x=14, linestyle='dashed', color='gray')

plt.legend(bbox_to_anchor=(1,1))
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.tight_layout()

plt.savefig('fig/a_tanh_halved.png')

# <codecell>
from scipy.stats import ortho_group as ortho_group

x = np.random.randn(4, 1) / np.sqrt(4)
M = ortho_group.rvs(4)

xs = [x]
for _ in range(100):
    x = M @ x + np.random.randn(*x.shape) * 0.1
    xs.append(x)

xs = np.array(xs)
norms = np.linalg.norm(xs, axis=1)


plt.plot(norms, 'o--')

# <codecell>
from scipy.stats import norm, cauchy
fac = 0.5

xs = np.linspace(-6 * fac, 6 * fac, 50)

plt.plot(norm.pdf(xs, scale=2 * fac))
plt.plot(cauchy.pdf(xs, scale=2 * fac))

# <codecell>
df = collate_dfs('remote/6_ac/generalize/')
df

# # <codecell>
# row = df.iloc[1]
# task = row['train_task']
# xs = next(task)

# A = xs[:,:task.n_dims,task.n_obs_dims:]
# C = xs[:,task.n_dims:(task.n_dims+task.n_obs_dims),task.n_obs_dims:]
# xs_vals = xs[:,(task.n_dims + task.n_obs_dims):, :task.n_obs_dims]

# xs_k = xs_vals
# pred_k, true_mse = pred_kalman(xs_k, task, A=A, C=C)
# xs_k = xs_k[:,1:]
# pred_k = pred_k[:,:-1]

# kalman_mse = ((xs_k - pred_k)**2).mean(axis=(0, -1))


# <codecell>
skip_idx = 0

def extract_plot_vals(row):
    time_len = len(row['info']['pred_mse'])

    task = row['train_task']
    xs = next(task)

    A = xs[:,:task.n_dims,task.n_obs_dims:]
    C = xs[:,task.n_dims:(task.n_dims+task.n_obs_dims),task.n_obs_dims:]
    xs_vals = xs[:,(task.n_dims + task.n_obs_dims):, :task.n_obs_dims]

    xs_k = xs_vals
    pred_k, kalman_true_mse = pred_kalman(xs_k, task, A=A, C=C)
    xs_k = xs_k[:,1:]
    pred_k = pred_k[:,:-1]

    kalman_mse = ((xs_k - pred_k)**2).mean(axis=(0, -1))

    return pd.Series([
        row['name'],
        row['config']['n_layers'],
        row['config']['n_mlp_layers'],
        row['config']['n_heads'],
        row['config']['pos_emb'],
        row['train_task'].n_snaps if row['train_task'].n_snaps is not None else 0,
        row['train_task'].t_noise,
        row['train_task'].n_obs_dims,
        row['info']['length'],
        row['info']['pred_mse'][skip_idx:],
        row['info']['zero_mse'][skip_idx:],
        kalman_mse[skip_idx:],
        kalman_true_mse[skip_idx:-1],
        np.arange(time_len)[skip_idx:]
    ], index=['name', 'n_layers', 'n_mlp_layers', 'n_heads', 'pos_emb', 'n_snaps', 'noise', 'n_obs_dims', 'length', 'pred_mse', 'zero_mse', 'kalman_mse', 'kalman_true_mse', 'time'])

tqdm.pandas(desc='kalman read')

plot_df = df.progress_apply(extract_plot_vals, axis=1)
plot_df

# <codecell>
plot_df = plot_df \
            .reset_index(drop=True) \
            .explode(['pred_mse', 'zero_mse', 'kalman_mse', 'kalman_true_mse', 'time']) \
            .melt(id_vars=['name', 'n_layers', 'n_mlp_layers', 'n_heads', 'pos_emb', 'n_snaps', 'noise', 'n_obs_dims', 'length', 'time'], var_name='mse_type', value_name='mse')
plot_df['mse'] = plot_df['mse'].astype(float)

plot_df

# <codecell>
mdf = plot_df.copy()

mdf = mdf[
    (mdf['n_snaps'] == 0)
    & (mdf['noise'] == 0.1)
    & (mdf['n_obs_dims'] == 16)
    & (mdf['pos_emb'] == False)
    & (mdf['length'] == 64)
]

gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', col='n_layers', row='n_heads', kind='line', marker='o', height=3, aspect=1.5, alpha=0.5)
# gs.set(yscale='log')

plt.savefig('fig/ac_big_sweep_nope_lin_scale.png')

# <codecell>
mdf = plot_df.copy()

mdf = mdf[
    (mdf['noise'] == 0.1)
    & (mdf['n_heads'] == 4)
    & (mdf['n_layers'] == 4)
    & (mdf['n_obs_dims'] == 16)
]

gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', col='pos_emb', row='length', kind='line', marker='o', height=3, aspect=1.5, alpha=0.5, facet_kws={'sharey': False})
# gs.set(yscale='log')

plt.savefig('fig/ac_big_sweep_pe_length_lin_scale.png')

