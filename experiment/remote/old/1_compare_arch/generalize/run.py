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
from model.transformer import TransformerConfig
from task.function import SameDifferent

run_id = new_seed()
print('RUN ID', run_id)

run_split = 3

train_iters = 25_000
n_vocab = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_dims = [2, 4, 8, 16, 32, 64, 128]

n_hidden = 512

### START TEST CONFIGS
# run_split = 1

# train_iters = 250
# n_vocab = [4]
# n_dims = [2]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for v in n_vocab:
    for d in n_dims:
        params = {'n_symbols': v, 'n_dims': d}
        
        all_cases.extend([
            Case(f'MLP', 
                MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden),
                train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(n_symbols=v, n_dims=d),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)),

            Case(f'Transformer (no MLP)', 
                TransformerConfig(n_out=1, n_layers=1, n_hidden=n_hidden, n_heads=2, 
                                  pos_emb=False, layer_norm=False, residual_connections=False, n_mlp_layers=0),
                train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(n_symbols=v, n_dims=d),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)),

            Case(f'Transformer (w/ MLP)', 
                TransformerConfig(n_out=1, n_layers=1, n_hidden=n_hidden, n_heads=2, 
                                  pos_emb=False, layer_norm=False, residual_connections=False, n_mlp_layers=2),
                train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(n_symbols=v, n_dims=d),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)),
        ])


all_cases = split_cases(all_cases, run_split)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

train_tasks = [c.train_task for c in all_cases]
test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=train_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.info['flops'] = case.get_flops()
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
