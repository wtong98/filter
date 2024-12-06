"""
Match task accuracies
"""

# <codecell>
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.mlp import MlpConfig 
from task.same_different import SameDifferent

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 50_000
n_vocab = [8, 64, 256]
log10_gs = np.linspace(-5, 3, num=9)
n_hiddens = [32, 64, 128, 256, 512, 1024, 4096, 8192]
n_dims = [32, 64, 128, 256, 512]
sig2s = [0, 0.25, 0.5, 1]

base_lr = 10
noise_scale = 1

n_layers = 1

### START TEST CONFIGS
# run_split = 1

# train_iters = 100
# n_vocab = [64]
# log10_gs = [0]
# n_hiddens = [128]
# n_dims = [64]
# sig2s = [0]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for sig2, log10_gamma0, d, n_hidden, v in itertools.product(sig2s, log10_gs, n_dims, n_hiddens, n_vocab):
    noise = sig2 * noise_scale

    gamma0 = 10**log10_gamma0
    gamma = gamma0 * np.sqrt(n_hidden)

    if gamma0 < 1:
        lr = gamma0**2 * base_lr
    else:
        lr = gamma0 * base_lr

    all_cases.append(
        Case(rf'$\gamma_0=10^{ {log10_gamma0} }$',
                MlpConfig(mup_scale=True, n_out=1, n_layers=1, n_hidden=n_hidden, use_bias=False),
                train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce',
                            'optim': optax.sgd, 'lr': lr, 'gamma': gamma},
                train_task=SameDifferent(n_symbols=v, n_dims=d, noise=noise),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024, noise=noise),
                info={'log10_gamma0': log10_gamma0,
                        'sig2': sig2})
    )

all_cases = split_cases(all_cases, run_split)
print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

train_tasks = [c.train_task for c in all_cases]
test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=train_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.state = None
    # case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
