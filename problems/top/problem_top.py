import os
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.beam_search import beam_search
from problems.top.state_top import StateTOP


class TOP(object):
    NAME = 'top'  # Team Orienteering Problem

    @staticmethod
    def get_costs(dataset, pi):

        # Index of end depot
        end_ids = dataset['loc'].shape[1] + 1
        end_depot = 'depot2' if 'depot2' in dataset else 'depot'

        # Check if tour consists in going to the end depot
        if pi.size(1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == end_ids).all(), "If all length 1 tours, they should be equal to the end depot index"
            # Return
            return torch.zeros(pi[:, 0, :].size(), dtype=torch.float, device=pi.device), None

        # Make sure each node visited once at most by only one agent (except for end depot)
        sorted_pi = pi.reshape((pi.shape[0], -1)).data.sort(1)[0]
        assert ((sorted_pi[:, 1:] == end_ids) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"

        # Prize and loc info
        prize_with_depot = torch.cat((torch.zeros_like(dataset['prize'][:, :1]), dataset['prize']), 1)
        prize_with_depot = torch.cat((prize_with_depot, torch.zeros_like(dataset['prize'][:, :1])), 1)
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        loc_with_depot = torch.cat((loc_with_depot, dataset[end_depot][:, None, :]), 1)

        # Calculate cost of each agent
        p = tuple()
        for k in range(pi.size(2)):

            # Prize collected
            p = p + (prize_with_depot.gather(1, pi[..., k])[:, :, None], )

            # Gather dataset in order of tour
            d = loc_with_depot.gather(1, pi[..., k][..., None].expand(*pi[..., k].size(), loc_with_depot.size(-1)))

            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
                + (d[:, 0] - dataset['depot']).norm(p=2, dim=-1)  # Depot to first
                + (d[:, -1] - dataset[end_depot]).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
            )
            assert (length <= dataset['max_length'] + 1e-5).all(), \
                "Max length exceeded by {}".format((length - dataset['max_length']).max())

        # We want to maximize total prize but code minimizes so return negative
        p = torch.cat(p, dim=2)
        return -p.sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TOPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTOP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TOP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, prize_type, max_length=2, num_depots=1):

    loc = torch.FloatTensor(size, 2).uniform_(0, 1)
    depot = torch.FloatTensor(2).uniform_(0, 1)

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = torch.ones(size)
    elif prize_type == 'unif':
        prize = (1 + torch.randint(0, 100, size=(size, ))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    # Output dataset
    dictionary = {'loc': loc, 'prize': prize, 'depot': depot, 'max_length': torch.tensor(max_length)}

    # End depot is different from start depot
    if num_depots == 2:
        depot2 = torch.FloatTensor(2).uniform_(0, 1)
        dictionary['depot2'] = depot2
    return dictionary


class TOPDataset(Dataset):

    def __init__(self, filename=None, size=20, num_samples=1000000, offset=0, distribution='const', num_depots=1,
                 max_length=2, **kwargs):
        super(TOPDataset, self).__init__()
        assert distribution is not None, "Data distribution must be specified for TOP"

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                print('Loading dataset...')
                data = pickle.load(f)
                if num_depots == 1:
                    self.data = [
                        {
                            'loc': torch.FloatTensor(loc),
                            'prize': torch.FloatTensor(prize),
                            'depot': torch.FloatTensor(depot),
                            'max_length': torch.tensor(length)
                        }
                        for depot, loc, prize, length in tqdm(data[offset:offset + num_samples])
                    ]
                else:
                    assert num_depots == 2, 'Number of depots has to be either 1 or 2.'
                    self.data = [
                        {
                            'loc': torch.FloatTensor(loc),
                            'prize': torch.FloatTensor(prize),
                            'depot': torch.FloatTensor(depot),
                            'max_length': torch.tensor(length),
                            'depot2': torch.FloatTensor(depot2)
                        }
                        for depot, loc, prize, length, depot2 in tqdm(data[offset:offset + num_samples])
                    ]
        else:
            print('Generating dataset...')
            self.data = [
                generate_instance(size, distribution, num_depots=num_depots, max_length=max_length)
                for _ in tqdm(range(num_samples))
            ]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
