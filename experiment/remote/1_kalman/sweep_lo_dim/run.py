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

run_split = 1

train_iters = 25_000
n_layers = [1]
n_widths = [512]
n_heads = [1]

n_tasks = [1, None]

n_dims = 128
n_obs_dims = [128, 64, 32, 16]

length = 16

### START TEST CONFIGS
# n_dims = 16
# length = 8

# run_split = 1

# train_iters = 50
# n_layers = [1]
# n_widths = [512]
# n_heads = [1]

# n_tasks = [1]
# n_obs_dims = [8]
### END TEST CONFIGS

all_cases = []

for n_head, n_width, n_layer, n_task, n_obs_d in itertools.product(n_heads, n_widths, n_layers, n_tasks, n_obs_dims):
    seed = new_seed()
    all_cases.extend([
        Case('Transformer',
            TransformerConfig(n_layers=n_layer,
                            n_hidden=n_width,
                            pos_emb=True,
                            n_mlp_layers=2,
                            n_heads=n_head,
                            layer_norm=True,
                            residual_connections=True,
                            freeze_emb=False,
                            return_final_logits_only=False,
                            n_out=n_obs_d),
            train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000},
            train_task=KalmanFilterTask(length=length, n_dims=n_dims, n_obs_dims=n_obs_d, t_noise=0.25, o_noise=0.25, seed=seed, n_tasks=n_task),
            test_task=KalmanFilterTask(length=length, n_dims=n_dims, n_obs_dims=n_obs_d, t_noise=0.25, o_noise=0.25, batch_size=1024, seed=seed, n_tasks=n_task),
        )
    ])

all_cases = split_cases(all_cases, run_split)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=test_tasks)

for case in all_cases:
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
