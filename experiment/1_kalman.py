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
from task.filter import KalmanFilterTask

# <codecell>
df = collate_dfs('remote/1_kalman/generalize')
df

# <codecell>
def extract_plot_vals(row):
    time_len = len(row['info']['pred_mse'])
    return pd.Series([
        row['name'],
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['config']['n_heads'],
        row['info']['pred_mse'],
        row['info']['naive_mse'],
        row['info']['zero_mse'],
        np.arange(time_len)
    ], index=['name', 'n_layers', 'n_hidden', 'n_heads', 'pred_mse', 'naive_mse', 'zero_mse', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \
            .explode(['pred_mse', 'naive_mse', 'zero_mse', 'time']) \
            .melt(id_vars=['name', 'n_layers', 'n_hidden', 'n_heads', 'time'], var_name='mse_type', value_name='mse')
plot_df['mse'] = plot_df['mse'].astype(float)

# <codecell>
mdf = plot_df[plot_df['n_hidden'] == 512]

gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', row='n_layers', col='n_heads', kind='line', marker='o', alpha=0.5, height=3)
gs.set(yscale='log')

plt.savefig('fig/filter.png')

# <codecell>
length = 16
n_dims = 32

train_task = KalmanFilterTask(length=length, n_dims=n_dims, t_noise=1, o_noise=1)
test_task = KalmanFilterTask(length=length, n_dims=n_dims, t_noise=1, o_noise=1)


config = TransformerConfig(n_layers=4,
                           n_hidden=1024,
                           pos_emb=True,
                           n_mlp_layers=2,
                           n_heads=4,
                           layer_norm=True,
                           residual_connections=True,
                           freeze_emb=False,
                           return_final_logits_only=False,
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
xs = next(train_task)
pred = state.apply_fn({'params': state.params}, xs)


# xs = np.linalg.norm(xs, axis=-1)
# pred = np.linalg.norm(pred, axis=-1)

pred_naive = xs[:,:-1]
xs = xs[:,1:]
pred = pred[:,:-1]

pred_mse = ((xs - pred)**2).mean(axis=(0, -1))
naive_mse = ((xs - pred_naive)**2).mean(axis=(0, -1))
zero_mse = (xs**2).mean(axis=(0, -1))

plt.plot(pred_mse, '--o', label='pred')
plt.plot(naive_mse, '--o', label='naive')
plt.plot(zero_mse, '--o', label='zero')

plt.legend()
plt.yscale('log')


# <codecell>
plt.plot(xs[0].mean(axis=-1), '--o')
plt.plot(pred[0].mean(axis=-1), '--o')
