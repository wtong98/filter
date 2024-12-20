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

run_split = 9

train_iters = 5_000
# n_layers = [1, 2, 4]
n_layers = [1, 2, 4]
n_widths = [2048]
# n_heads = [1, 2]
n_heads = [1, 2, 4]

noises = [0.1]
lengths = [16]
max_svals = [1]

pos_emb = [False]
n_tasks = [1]
max_svals = [1]

n_dims = 32

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
# max_svals = [1]
### END TEST CONFIGS

all_cases = []

for p_emb, n_task, max_sval, noise, n_head, n_width, n_layer, length in itertools.product(pos_emb, n_tasks, max_svals, noises, n_heads, n_widths, n_layers, lengths):
    seed = new_seed()
    all_cases.extend([
        Case('Transformer',
            TransformerConfig(n_layers=n_layer,
                            n_hidden=n_width,
                            pos_emb=p_emb,
                            n_mlp_layers=0,
                            n_heads=n_head,
                            layer_norm=False,
                            residual_connections=False,
                            freeze_emb=False,
                            return_final_logits_only=False,
                            n_out=n_dims),
            train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000},
            train_task=KalmanFilterTask(length=length, n_tasks=n_task, n_dims=n_dims, max_sval=max_sval, t_noise=noise, o_noise=noise, seed=seed),
            test_task=KalmanFilterTask(length=length, n_tasks=n_task, n_dims=n_dims, max_sval=max_sval, t_noise=noise, o_noise=noise, batch_size=1024, seed=seed),
        )
    ])

all_cases = split_cases(all_cases, run_split)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

test_tasks = [c.test_task for c in all_cases]
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
