"""Prepare cifar100 task"""

# <codecell>
# from collections import defaultdict
# import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

# <codecell>
def load_data():
    dataset_path = Path(__file__).parent / Path('dataset/cifar100')

    with open(dataset_path / 'train', 'rb') as fp:
        train_set = pickle.load(fp, encoding='bytes')

    with open(dataset_path / 'test', 'rb') as fp:
        test_set = pickle.load(fp, encoding='bytes')

    with open(dataset_path / 'meta', 'rb') as fp:
        meta_set = pickle.load(fp)


    train_labs = train_set[b'fine_labels']
    train_data = train_set[b'data']

    test_labs = test_set[b'fine_labels']
    test_data = test_set[b'data']

    all_labs = train_labs + test_labs
    all_data = np.concatenate((train_data, test_data), axis=0)
    all_data = (all_data - all_data.mean()) / all_data.std()

    label_names = meta_set['fine_label_names']

    # label_to_idxs = defaultdict(list)
    # for i, lab in enumerate(all_labs):
    #     label_to_idxs[lab].append(i)
    
    return {
        'data': all_data,
        'labels': all_labs,
        'names': label_names
    }

# %%
# label_to_idxs[85]


# # <codecell>
# ims = all_data.reshape(-1, 32, 32, 3, order='F')

# plt.imshow(ims[2])


