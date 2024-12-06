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

length = 16
n_dims = 32

train_task = KalmanFilterTask(length=length, n_dims=n_dims)
test_task = KalmanFilterTask(length=length, n_dims=n_dims)


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

pred_mse = np.mean((xs - pred)**2, axis=[0, -1])
naive_mse = np.mean((xs - pred_naive)**2, axis=[0, -1])
zero_mse = np.mean(xs**2, axis=0)

plt.plot(pred_mse, '--o')
plt.plot(naive_mse, '--o')
plt.plot(zero_mse, '--o')

# plt.yscale('log')


# <codecell>

plt.plot(xs[0], '--o')
plt.plot(pred[0], '--o')
