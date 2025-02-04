"""Modeling non-Markovian sequences"""

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


config = TransformerConfig(n_layers=1,
                           n_hidden=n_hidden,
                           pos_emb=False,
                        #    pos_emb_concat_dim=n_hidden,
                           n_mlp_layers=0,
                           n_heads=n_heads,
                           layer_norm=False,
                           residual_connections=False,
                           freeze_emb=False,
                           return_final_logits_only=False,
                        #    use_simple_att=True,
                           n_out=n_obs_dims)

config_pe = TransformerConfig(n_layers=1,
                           n_hidden=n_hidden,
                           pos_emb=True,
                        #    pos_emb_concat_dim=n_hidden,
                           n_mlp_layers=0,
                           n_heads=n_heads,
                           layer_norm=False,
                           residual_connections=False,
                           freeze_emb=False,
                           return_final_logits_only=False,
                        #    use_simple_att=True,
                           n_out=n_obs_dims)

config_1h = TransformerConfig(n_layers=2,
                           n_hidden=n_hidden,
                           pos_emb=False,
                        #    pos_emb_concat_dim=n_hidden,
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

state_pe, hist = train(config_pe,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_every=1000,
                    train_iters=5_000, 
                    seed=None)

state_1h, hist = train(config_1h,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_every=1000,
                    train_iters=5_000, 
                    seed=None)

# <codecell>
# train_task = KalmanFilterTask(
#     length=length, 
#     n_obs_dims=1, 
#     n_dims=3, 
#     t_noise=noise, o_noise=noise, seed=seed, max_sval=1)

def pred_kalman(xs, task, zs_init=None, return_mat=False):
    I = np.eye(task.n_dims)
    Io = np.eye(task.n_obs_dims)

    A = task.t_mat
    C = task.o_mat
    S_u = I * task.t_noise / task.n_dims
    S_w = Io * task.o_noise / task.n_obs_dims

    ba = np.zeros((task.n_dims, task.batch_size)) if zs_init is None else zs_init
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

        true_mse = np.trace(C @ Sa @ C.T + S_w) / task.n_obs_dims
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

train_task.n_tasks = 1
train_task.batch_size = 4096
train_task.length = 64
train_task.cheat_mode = True

xs, zs_init = next(train_task)
zs_init = zs_init.squeeze().T
zs_init = None

pred = state.apply_fn({'params': state.params}, xs)
pred_pe = state_pe.apply_fn({'params': state_pe.params}, xs)
pred_1h = state_1h.apply_fn({'params': state_1h.params}, xs)

pred_naive = xs[:,:-1]
pred = pred[:,:-1]
pred_pe = pred_pe[:,:-1]
pred_1h = pred_1h[:,:-1]

train_task.n_tasks = 1
xs_k = xs
pred_k, kalman_mat = pred_kalman(xs_k, train_task, zs_init=zs_init, return_mat=True)
xs_k = xs_k[:,1:]
pred_k = pred_k[:,:-1]

xs = xs[:,1:]

pred_mse = ((xs - pred)**2).mean(axis=(0, -1))
pred_pe_mse = ((xs - pred_pe)**2).mean(axis=(0, -1))
pred_1h_mse = ((xs - pred_1h)**2).mean(axis=(0, -1))
naive_mse = ((xs - pred_naive)**2).mean(axis=(0, -1))
zero_mse = (xs**2).mean(axis=(0, -1))
kalman_mse = ((xs_k - pred_k)**2).mean(axis=(0, -1))

plt.plot(pred_mse[:], '--o', label='Transformer (NoPE, $H=4$)', alpha=0.7, color='C9')
plt.plot(pred_1h_mse[:], '--o', label='Transformer (NoPE, $H=1, \ell=2$)', alpha=0.7, color='C0')
plt.plot(pred_pe_mse[:], '--o', label='Transformer (PE, $H=4$)', alpha=0.7, color='mediumseagreen')
# plt.plot(naive_mse, '--o', label='naive', alpha=0.7)
plt.plot(kalman_mse[:], '--o', label='Kalman', alpha=0.7, color='C1')
plt.plot(zero_mse[:], '--o', label='Zero', alpha=0.7, color='C8')
# plt.plot(true_mse[1:], label='kalman_true', alpha=0.7)

plt.axvline(x=14, linestyle='dashed', color='gray')

plt.legend(bbox_to_anchor=(1,1))
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.tight_layout()

plt.savefig('fig/extrapolation_non_markov.png', bbox_inches='tight')

# <codecell>
plt.plot(pred[0,:,0], 'o--', alpha=0.7, label='Transformer')
plt.plot(pred_k[0,:,0], 'o--', alpha=0.7, label='KF')
plt.plot(xs[0,:,0], 'o--', alpha=0.7, label='True')
plt.legend()

plt.savefig('fig/example_run.png')

# <codecell>
kalman_mat = kalman_mat[:16,:16]
plt.imshow(np.abs(kalman_mat))
plt.colorbar()
plt.axis('off')
plt.savefig('fig/final_report/kalman_mat_non_markov.svg')

# <codecell>
train_task.n_tasks = 1
train_task.batch_size = 128
train_task.length = 16

xs = next(train_task)
pred, info = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

att = info['intermediates']['TransformerBlock_0']['MultiHeadDotProductAttention_0']['attention_weights'][0].squeeze()

fig, axs = plt.subplots(1, 4, figsize=(12, 3))

for ax, a in zip(axs.ravel(), att[0]):
    im = ax.imshow(a)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('fig/attention_no_pos_enc.png')

# <codecell>
jax.tree.map(lambda x: x.shape, state.params)

# <codecell>
pe = sin_init(None, (1, length, n_hidden), None).squeeze()
pe = pe[:length]

E = state.params['Dense_0']['kernel']
D = state.params['Dense_1']['kernel']
V = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
O = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()

Ms = np.stack([(E @ V[:,i] @ O[i] @ D) for i in range(n_heads)])
pes = np.stack([(pe @ V[:,i] @ O[i] @ D) for i in range(n_heads)])

big_mats = att * Ms
big_mats = np.sum(big_mats, axis=1)

pes = att @ pes
pes = np.sum(pes, axis=1)

# ps = big_mats @ xs + pes
ps = big_mats @ xs

print(pred[0,:10])
print(ps[0,:10])

np.mean((pred - ps)**2)

# <codecell>
plt.imshow(np.abs(big_mats[0]))
plt.colorbar()
plt.savefig('fig/transformer_mat_no_pos_enc.png')

# <codecell>
idx = 0

dists = []

big_d = (xs.shape[1] - 1) * xs.shape[2]
big_mats_acc = np.zeros((big_d, big_d))
all_ms = []

for idx in tqdm(range(xs.shape[0])[:10]):
    x = xs[idx][:-1]
    A = att[idx]

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

    # x = x.reshape(-1, 1)
    # x_pred = big_mat @ x
    # x_pred = x_pred.reshape(A.shape[0], -1)

    k_preds, kalman_mat = pred_kalman(xs, train_task, return_mat=True)

    big_mat_vals = big_mat[~np.isclose(kalman_mat, 0)]
    kal_mat_vals = kalman_mat[~np.isclose(kalman_mat, 0)]

    m = np.linalg.lstsq(big_mat_vals[:,None], kal_mat_vals)[0]
    all_ms.append(m)

    dists.append(np.mean((big_mat - kalman_mat)**2))

# <codecell>
# plt.imshow(big_mat)
# plt.imshow(A)
# plt.colorbar()

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
        A = att[idx]

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
plt.savefig('fig/transformer_v_kalman_vary_att_small_obs.png')

# <codecell>
plt.title('Abs Kalman Matrix')
plt.imshow(np.abs(kalman_mat))
plt.colorbar()
plt.savefig('fig/kalman_mat_small_obs.png')
