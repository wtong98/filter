"""Experimenting with Kalman filter"""

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

# <codecell>
df = collate_dfs('remote/1_kalman/generalize')
df

# <codecell>
def extract_plot_vals(row):
    time_len = len(row['info']['pred_mse'])

    task = row['test_task']
    task.n_tasks = 1
    xs = next(task)
    kalman_preds = pred_kalman(xs, task)

    xs = xs[:,1:]
    kalman_preds = kalman_preds[:,:-1]
    kalman_mse = ((xs - kalman_preds)**2).mean(axis=(0, -1))

    return pd.Series([
        row['name'],
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['config']['n_heads'],
        row['train_task'].t_noise,
        row['train_task'].length,
        row['train_task'].max_sval,
        row['info']['pred_mse'],
        row['info']['naive_mse'],
        row['info']['zero_mse'],
        kalman_mse,
        np.arange(time_len)
    ], index=['name', 'n_layers', 'n_hidden', 'n_heads', 'noise', 'length', 'max_sval', 'pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \
            .explode(['pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'time']) \
            .melt(id_vars=['name', 'n_layers', 'n_hidden', 'n_heads', 'noise', 'length', 'max_sval', 'time'], var_name='mse_type', value_name='mse')
plot_df['mse'] = plot_df['mse'].astype(float)

plot_df

# <codecell>
mdf = plot_df[plot_df['n_hidden'] == 2048]
# mdf = mdf[(mdf['mse_type'] != 'naive_mse')
#           & (mdf['noise'] == 0.1)
#           & (mdf['length'] == 16)
#           & (mdf['max_sval'] == 2)]

mdf = mdf[(mdf['mse_type'] != 'naive_mse')
          & (mdf['n_layers'] == 2)
          & (mdf['n_heads'] == 2)
          & (mdf['length'] == 16)]

gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', row='noise', col='max_sval', kind='line', marker='o', alpha=0.5, height=3, facet_kws={'sharey': False})
gs.set(yscale='log')

# plt.savefig('fig/filter_noise_var_mat_nhead2_nlayer2.png')


# <codecell>
df = collate_dfs('remote/1_kalman/sweep_lo_dim')
df

# <codecell>
def extract_plot_vals(row):
    time_len = len(row['info']['pred_mse'])

    task = row['test_task']
    task.n_tasks = 1
    xs = next(task)
    kalman_preds = pred_kalman(xs, task)

    xs = xs[:,1:]
    kalman_preds = kalman_preds[:,:-1]
    kalman_mse = ((xs - kalman_preds)**2).mean(axis=(0, -1))

    return pd.Series([
        row['name'],
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['config']['n_heads'],
        row['train_task'].n_obs_dims,
        row['train_task'].n_tasks,
        row['info']['pred_mse'],
        row['info']['naive_mse'],
        row['info']['zero_mse'],
        kalman_mse,
        np.arange(time_len)
    ], index=['name', 'n_layers', 'n_hidden', 'n_heads', 'n_obs_dims', 'n_tasks', 'pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \
            .explode(['pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'time']) \
            .melt(id_vars=['name', 'n_layers', 'n_hidden', 'n_heads', 'n_obs_dims', 'n_tasks', 'time'], var_name='mse_type', value_name='mse')
plot_df['mse'] = plot_df['mse'].astype(float)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['n_tasks'] == 1) & (mdf['mse_type'] != 'naive_mse')] 


gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', col='n_obs_dims', kind='line', marker='o', alpha=0.5, height=3, facet_kws={'sharey': False})
gs.set(yscale='log')

plt.savefig('fig/low_obs_dim_diff_y.png')

# <codecell>
task = KalmanFilterTask(length=16, n_dims=32, t_noise=0.25, o_noise=0.25)
task = df.iloc[0].test_task

xs = next(task)
preds = pred_kalman(xs, task)

xs = xs[:,1:]
preds = preds[:,:-1]

k_mse = ((xs - preds)**2).mean(axis=(0, -1))
z_mse = ((xs)**2).mean(axis=(0, -1))

print(k_mse)
print(z_mse)

# <codecell>
length = 16
n_dims = 4

noise = 0.25

seed = new_seed()
train_task = KalmanFilterTask(length=length, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed)
test_task = KalmanFilterTask(length=length, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed)


config = TransformerConfig(n_layers=1,
                           n_hidden=512,
                           pos_emb=False,
                           n_mlp_layers=0,
                           n_heads=1,
                           layer_norm=False,
                           residual_connections=False,
                           freeze_emb=False,
                           return_final_logits_only=False,
                        #    use_simple_att=True,
                           n_out=n_dims)


# xs = next(train_task)
# model = config.to_model()
# params = model.init(jax.random.PRNGKey(3), xs)['params']

# out = model.apply({'params': params}, xs)
# out.shape

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_every=500,
                    train_iters=10_000, 
                    seed=None)

# <codecell>
train_task.n_tasks = 1
train_task.batch_size = 1024
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
pred_k = pred_kalman(xs_k, train_task)
xs_k = xs_k[:,1:]
pred_k = pred_k[:,:-1]

xs = xs[:,1:]

pred_mse = ((xs - pred)**2).mean(axis=(0, -1))
naive_mse = ((xs - pred_naive)**2).mean(axis=(0, -1))
zero_mse = (xs**2).mean(axis=(0, -1))
kalman_mse = ((xs_k - pred_k)**2).mean(axis=(0, -1))

# normalize by magnitude
pred_mse /= zero_mse
kalman_mse /= zero_mse
zero_mse /= zero_mse

plt.plot(pred_mse, '--o', label='pred', alpha=0.7)
# plt.plot(naive_mse, '--o', label='naive', alpha=0.7)
plt.plot(zero_mse, '--o', label='zero', alpha=0.7)
plt.plot(kalman_mse, '--o', label='kalman', alpha=0.7)

plt.axvline(x=14, linestyle='dashed', color='gray')

plt.legend()
# plt.yscale('log')
plt.xlabel('time')
plt.ylabel('mse')
plt.tight_layout()

plt.savefig('fig/extrapolation_nope_no_mlp_softmax_att_dim4.png')

# <codecell>
train_task.n_tasks = 1
train_task.batch_size = 1024
train_task.length = 16

xs = next(train_task)
pred, info = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

att = info['intermediates']['TransformerBlock_0']['MultiHeadDotProductAttention_0']['attention_weights'][0].squeeze()

fig, axs = plt.subplots(1, 4, figsize=(12, 3))

for ax, a in zip(axs.ravel(), att):
    im = ax.imshow(a)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('fig/attention.png')
# <codecell>
plt.plot(xs[0].mean(axis=-1), '--o')
plt.plot(pred[0].mean(axis=-1), '--o')
