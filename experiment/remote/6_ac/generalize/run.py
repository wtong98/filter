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

from model.transformer import TransformerConfig
from task.filter import KalmanFilterTask

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 20_000
# n_layers = [1, 2, 4]
n_layers = [1, 2, 4]
n_widths = [512]
# n_heads = [1, 2]
n_heads = [1, 2]
n_mlp_layers = [0, 2]

noises = [0.001, 0.01, 0.1]
lengths = [16]

pos_emb = [False]
n_tasks = [None]

n_snaps = [None, 4, 16]
n_obs_dims = [2, 4]
n_dims = 4

### START TEST CONFIGS
# n_dims = 16
# length = 8

# run_split = 1

# train_iters = 50
# n_layers = [1]
# n_widths = [512]
# n_heads = [1]

# noises = [0.001]
# lengths = [16]

# n_snaps = [None]
# n_obs_dims = [4]
### END TEST CONFIGS

all_cases = []

for n_task, noise, n_head, n_width, n_layer, n_mlp_layer, length, n_snap, n_obs_dim \
    in itertools.product(n_tasks, noises, n_heads, n_widths, n_layers, n_mlp_layers, lengths, n_snaps, n_obs_dims):

    seed = new_seed()
    all_cases.extend([
        Case('Transformer',
            TransformerConfig(n_layers=n_layer,
                            n_hidden=n_width,
                            pos_emb=False,
                            n_mlp_layers=n_mlp_layer,
                            n_heads=n_head,
                            layer_norm=False,
                            residual_connections=True,
                            freeze_emb=False,
                            return_final_logits_only=False,
                            n_out=n_dims),
            train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000},
            train_task=KalmanFilterTask(length=length, 
                                        n_tasks=n_task, 
                                        n_dims=n_dims, 
                                        n_obs_dims=n_obs_dim,
                                        t_noise=noise, 
                                        o_noise=noise,
                                        mode='ac',
                                        n_snaps=n_snap,
                                        seed=seed)
        )
    ])

all_cases = split_cases(all_cases, run_split)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

test_tasks = [c.train_task for c in all_cases]
for task in test_tasks:
    task.length = 64

eval_cases(all_cases, eval_task=test_tasks)

for case in all_cases:
    # case.info['params'] = case.state.params
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
