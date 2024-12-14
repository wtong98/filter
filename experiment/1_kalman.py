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

set_theme()

# <codecell>
df = collate_dfs('remote/1_kalman/generalize/set4_good_sample')
df

# <codecell>
def extract_plot_vals(row):
    time_len = len(row['info']['pred_mse'])

    task = row['test_task']
    task.n_tasks = 1
    xs = next(task)
    kalman_preds, kalman_true_mse = pred_kalman(xs, task)

    kalman_true_mse = kalman_true_mse[:-1]

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
        kalman_true_mse,
        np.arange(time_len)
    ], index=['name', 'n_layers', 'n_hidden', 'n_heads', 'noise', 'length', 'max_sval', 'pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'kalman_true_mse', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \
            .explode(['pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'kalman_true_mse', 'time']) \
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
        #   & (mdf['n_layers'] == 2)
        #   & (mdf['n_heads'] == 2)
        #   & (mdf['length'] == 16)
          ]

gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', col='noise', kind='line', marker='o', alpha=0.5, height=3, facet_kws={'sharey': False})
gs.set(yscale='log')

# plt.savefig('fig/filter_noise_var_mat_nhead2_nlayer2.png')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['mse_type'] != 'naive_mse')
          & (mdf['noise'] == 0.1)
          & (mdf['time'] > 0)
          ]

mdf = mdf.replace({
    'pred_mse': 'Transformer',
    'zero_mse': 'Zero Predictor',
    'kalman_mse': 'Kalman (actual)',
    'kalman_true_mse': 'Kalman (theory)'
})
# <codecell>
g = sns.lineplot(mdf, x='time', y='mse', hue='mse_type', 
                 hue_order=['Transformer', 'Kalman (actual)', 'Zero Predictor'], 
                 palette=['C0', 'C1', 'C8'], 
                 marker='o', markersize=5, alpha=0.7)
g.set_yscale('log')
g.set_xlabel('Time')
g.set_ylabel('MSE')

g.axvline(x=14, linestyle='dashed', color='gray')

adf = mdf[mdf['mse_type'] == 'Kalman (theory)']
sns.lineplot(adf, x='time', y='mse', linewidth=1, ax=g, color='red', ci=None, alpha=0.7)

handles, labels = g.get_legend_handles_labels()
handles.insert(2, g.get_children()[10])
labels.insert(2, 'Kalman (theory)')

plt.text(11.8, 9e-3, 'train/test split', rotation=90, va='center', color='gray', fontsize=10)

plt.legend(handles, labels)

g.legend_.set_title(None)

g.figure.set_size_inches(5.5, 3.5)
g.figure.tight_layout()

sns.move_legend(g, "upper left", bbox_to_anchor=(0.45, 1.1))

plt.savefig('fig/transformer_vs_kalman_sample.png')

# <codecell>
### LARGE SWEEP
df = collate_dfs('remote/1_kalman/generalize')
df

# <codecell>
def extract_plot_vals(row):
    time_len = len(row['info']['pred_mse'])

    task = row['test_task']
    task.n_tasks = 1
    xs = next(task)
    kalman_preds, kalman_true_mse = pred_kalman(xs, task)

    kalman_true_mse = kalman_true_mse[:-1]

    xs = xs[:,1:]
    kalman_preds = kalman_preds[:,:-1]
    kalman_mse = ((xs - kalman_preds)**2).mean(axis=(0, -1))

    return pd.Series([
        row['name'],
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['config']['n_heads'],
        row['config']['pos_emb'],
        row['train_task'].t_noise,
        row['train_task'].length,
        row['train_task'].max_sval,
        row['train_task'].n_tasks,
        row['info']['pred_mse'],
        row['info']['naive_mse'],
        row['info']['zero_mse'],
        kalman_mse,
        kalman_true_mse,
        np.arange(time_len)
    ], index=['name', 'n_layers', 'n_hidden', 'n_heads', 'pos_emb', 'noise', 'length', 'max_sval', 'n_tasks', 'pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'kalman_true_mse', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \
            .explode(['pred_mse', 'naive_mse', 'zero_mse', 'kalman_mse', 'kalman_true_mse', 'time']) \
            .melt(id_vars=['name', 'n_layers', 'n_hidden', 'n_heads', 'pos_emb', 'noise', 'length', 'max_sval', 'n_tasks', 'time'], var_name='mse_type', value_name='mse')
plot_df['mse'] = plot_df['mse'].astype(float)

plot_df


# <codecell>
# n_head / n_depth sweep
mdf = plot_df.copy()

mdf = mdf[(mdf['mse_type'] != 'naive_mse')
          & (mdf['noise'] == 0.1)
          & (mdf['max_sval'] == 1)
          & (mdf['pos_emb'] == False)
          & (mdf['time'] > 0)
          & (mdf['n_layers'] != 4)
          ]

mdf = mdf.replace({
    'pred_mse': 'Transformer',
    'zero_mse': 'Zero Predictor',
    'kalman_mse': 'Kalman (actual)',
    'kalman_true_mse': 'Kalman (theory)'
})


gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', 
            row='n_layers',
            col='n_heads',
            kind='line', 
            marker='o', alpha=0.7, markersize=5,
            hue_order=['Transformer', 'Kalman (actual)', 'Kalman (theory)', 'Zero Predictor'],
            palette=['C0', 'C1', 'C3', 'C8'], 
            height=3, aspect=1.2)

gs.set(yscale='log')
gs.set_xlabels('Time')
gs.set_ylabels('MSE')
gs.set_titles("{row_name} Layers, {col_name} Heads")

gs.legend.set_title(None)
plt.savefig('fig/head_layer_sweep.png')


# <codecell>
# impact of noise

# NOTE: noise should be over higher values
mdf = plot_df.copy()

mdf = mdf[(mdf['mse_type'] != 'naive_mse')
          & (mdf['max_sval'] == 1)
          & (mdf['pos_emb'] == False)
          & (mdf['time'] > 0)
          & (mdf['n_layers'] == 1)
          & (mdf['n_heads'] == 1)
          ]

mdf = mdf.replace({
    'pred_mse': 'Transformer',
    'zero_mse': 'Zero Predictor',
    'kalman_mse': 'Kalman (actual)',
    'kalman_true_mse': 'Kalman (theory)'
})

gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', 
            col='noise',
            kind='line', 
            marker='o', alpha=0.7, markersize=5,
            hue_order=['Transformer', 'Kalman (actual)', 'Kalman (theory)', 'Zero Predictor'],
            palette=['C0', 'C1', 'C3', 'C8'], 
            height=3, aspect=1.2, facet_kws={'sharey': False})

gs.set(yscale='log')
gs.set_xlabels('Time')
gs.set_ylabels('MSE')
# gs.set_titles("{row_name} Layers, {col_name} Heads")

gs.legend.set_title(None)

# <codecell>
# Fixed vs. varying AC and max SV

# <codecell>
# impact of positional embeddings
mdf = plot_df.copy()

mdf = mdf[(mdf['mse_type'] != 'naive_mse')
          & (mdf['max_sval'] == 1)
          & (mdf['noise'] == 0.1)
          & (mdf['pos_emb'] == True)
          & (mdf['time'] > 0)
          & (mdf['n_layers'] == 1)
          & (mdf['n_heads'] == 1)
          ]

mdf = mdf.replace({
    'pred_mse': 'Transformer',
    'zero_mse': 'Zero Predictor',
    'kalman_mse': 'Kalman (actual)',
    'kalman_true_mse': 'Kalman (theory)'
})

g = sns.lineplot(mdf, x='time', y='mse', hue='mse_type', 
                 hue_order=['Transformer', 'Kalman (actual)', 'Zero Predictor'], 
                 palette=['C0', 'C1', 'C8'], 
                 marker='o', markersize=5, alpha=0.7)
g.set_yscale('log')
g.set_xlabel('Time')
g.set_ylabel('MSE')

g.axvline(x=14, linestyle='dashed', color='gray')

adf = mdf[mdf['mse_type'] == 'Kalman (theory)']
sns.lineplot(adf, x='time', y='mse', linewidth=1, ax=g, color='red', ci=None, alpha=0.7)

handles, labels = g.get_legend_handles_labels()
handles.insert(2, g.get_children()[10])
labels.insert(2, 'Kalman (theory)')

plt.text(11.8, 9e-3, 'train/test split', rotation=90, va='center', color='gray', fontsize=10)

plt.legend(handles, labels)

g.legend_.set_title(None)

g.figure.set_size_inches(5.5, 3.5)
g.figure.tight_layout()

sns.move_legend(g, "upper left", bbox_to_anchor=(0.45, 1.1))




# <codecell>
gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', 
            col='noise',
            kind='line', 
            marker='o', alpha=0.7, markersize=5,
            hue_order=['Transformer', 'Kalman (actual)', 'Kalman (theory)', 'Zero Predictor'],
            palette=['C0', 'C1', 'C3', 'C8'], 
            height=3, aspect=1.2, facet_kws={'sharey': False})

gs.set(yscale='log')
gs.set_xlabels('Time')
gs.set_ylabels('MSE')
# gs.set_titles("{row_name} Layers, {col_name} Heads")

gs.legend.set_title(None)





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


gs = sns.relplot(mdf, x='time', y='mse', hue='mse_type', 
                 col='n_obs_dims', 
                 kind='line', 
                 marker='o', 
                 alpha=0.5, 
                 height=3, 
                 facet_kws={'sharey': True})
gs.set(yscale='log')

# plt.savefig('fig/low_obs_dim_diff_y.png')

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
n_dims = 32
n_obs_dims = 32

noise = 0.1

seed = new_seed()
train_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)
test_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)


config = TransformerConfig(n_layers=1,
                           n_hidden=2048,
                           pos_emb=True,
                           n_mlp_layers=0,
                           n_heads=1,
                           layer_norm=False,
                           residual_connections=False,
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
                    test_every=500,
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

# plt.savefig('fig/extrapolation_nope_no_mlp_softmax_att_dim4.png')

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
# plt.savefig('fig/attention.png')

# <codecell>
jax.tree.map(lambda x: x.shape, state.params)

E = state.params['Dense_0']['kernel']
D = state.params['Dense_1']['kernel']
V = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
O = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()

M = (E @ V @ O @ D)

ps = xs @ M
ps.shape

att.shape

ps = jnp.einsum('bqk,bkd->...bqd', att, ps)

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
    A = att[idx]

    big_mat = []
    for i in range(A.shape[0] - 1):
        row = []
        for j in range(A.shape[1] - 1):
            row.append(A[i,j] * M.T)
        
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

for m_sel in all_ms:
    plt.plot(vals, m_sel * vals, color='thistle', alpha=0.1)

m = np.mean(all_ms)

plt.plot(vals, m * vals, color='red', alpha=0.5)
plt.scatter(big_mat_vals, kal_mat_vals, alpha=0.03, s=1)

plt.text(-0.17, 0.1, f'$r^2 = {r2:.2f}$', color='red', fontsize=10)

plt.xlabel('Transformer coefficients')
plt.ylabel('Kalman coefficients')

plt.tight_layout()
plt.savefig('fig/transformer_vs_kalman_reg_coeff.png')


# <codecell>
plt.hist(dists, bins=50)
plt.xlabel('Entry-wise MSE')
plt.ylabel('Count')

plt.tight_layout()


# <codecell>
big_mat_svals = np.linalg.svdvals(big_mat)
kal_mat_svals = np.linalg.svdvals(kalman_mat)

cutoff = 1e-3
n_cutoff = np.sum(kal_mat_svals < cutoff)

ranks = np.arange(len(kal_mat_svals))[:-n_cutoff]
ratios = (big_mat_svals / kal_mat_svals)[:-n_cutoff]

# <codecell>
vals = np.linspace(0, 480, 500)
plt.plot(vals, 0 * vals + 1, '--', color='gray')
plt.scatter(ranks, ratios, s=5, alpha=0.9)

plt.xlabel('SV rank')
plt.ylabel('Tf SV / Km SV')

plt.tight_layout()
plt.savefig('fig/transformer_vs_kalman_sv.png', bbox_inches='tight')


