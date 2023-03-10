import os
import math
import torch
import random
import pickle
import argparse
import numpy as np
from torch.utils.data import Dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def str2bool(v):
    """
    Transform string inputs into boolean.
    :param v: string input.
    :return: string input transformed to boolean.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def assign_colors(n):
    color = {k: [] for k in 'rgb'}
    for i in range(n):
        temp = {k: random.randint(0, 230) for k in 'rgb'}
        for k in temp:
            while 1:
                c = temp[k]
                t = set(j for j in range(c - 25, c + 25) if 0 <= j <= 230)
                if t.intersection(color[k]):
                    temp[k] = random.randint(0, 230)
                else:
                    break
            color[k].append(temp[k])
    return [(color['r'][i] / 256, color['g'][i] / 256, color['b'][i] / 256) for i in range(n)]


class ScenarioDataset(Dataset):
    def __init__(self, path, graph_size, padding):
        super(ScenarioDataset, self).__init__()
        self.path = path
        self.graph_size = graph_size
        self.samples = os.listdir(path)
        self.padding = padding

    def __getitem__(self, item):

        # Load file
        filename = os.path.join(self.path, str(item).zfill(self.padding) + '.pkl')
        with open(filename, 'rb') as f:
            sample = pickle.load(f)
        for k, v in sample.items():
            sample[k] = np.array(v, dtype=np.float32)
        return sample

    def __len__(self):
        return len(self.samples)
