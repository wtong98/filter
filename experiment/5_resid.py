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

noise = 0.001

# seed = new_seed()
seed = 3930462014
train_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)
test_task = KalmanFilterTask(length=length, n_obs_dims=n_obs_dims, n_tasks=1, n_dims=n_dims, t_noise=noise, o_noise=noise, seed=seed, max_sval=1)


config = TransformerConfig(n_layers=2,
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
