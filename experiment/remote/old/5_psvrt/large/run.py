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
from task.same_different import SameDifferentPentomino

run_id = new_seed()
print('RUN ID', run_id)

run_split = 1

ps = np.random.permutation(np.arange(18))

train_iters = 500_000
n_hidden = 512

n_widths = [2, 3, 4, 5, 6, 7, 8]
n_trains = [16]
gs = [0.01, 0.1, 1, 10]
base_lr = 0.1
blur = 0
random_blur = False
test_blur = 0

### START TEST CONFIGS
# n_widths = [2]
# train_iters = 1000
# n_hidden = 512

# n_trains = [16]
# gs = [1]
# base_lr = 0.1
### END TEST CONFIGS

all_cases = []
test_tasks = []

for n_train, n_width in itertools.product(n_trains, n_widths):
    train_ps = ps[:n_train]
    test_ps = ps[n_train:]

    all_cases.extend([
        Case(f'MLP (Adam)', 
            MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden),
            train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
            train_task=SameDifferentPentomino(ps=train_ps, blur=blur, random_blur=random_blur, width=n_width),
            test_task=SameDifferentPentomino(ps=test_ps, batch_size=1024, blur=test_blur, width=n_width)),
    ] + [
            Case(f'MLP (gamma={gamma})', 
                MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden, feature_learning_strength=gamma),
                train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce', 'optim': optax.sgd, 'lr': base_lr * gamma},
                train_task=SameDifferentPentomino(ps=train_ps, blur=blur, random_blur=random_blur, width=n_width),
                test_task=SameDifferentPentomino(ps=test_ps, batch_size=1024, blur=test_blur, width=n_width))
        for gamma in gs])


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

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
