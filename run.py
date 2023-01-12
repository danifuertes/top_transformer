#!/usr/bin/env python

import os
import json
import torch
import pprint as pp
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger  # pip install protobuf==3.20.*

from utils.reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils.train import train_epoch, validate, get_inner_model
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem
from nets.critic_network import CriticNetwork
from utils.options import get_options
from nets.gamma import GAMMA
from nets.gpn import GPN


def run(opts):

    # Save arguments so exact configuration can always be found
    if not opts.eval_only:
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

        # Optionally configure tensorboard
        tb_logger = None
        tb_dir = os.path.join(opts.save_dir, 'log_dir')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir, exist_ok=True)
        if not opts.no_tensorboard:
            tb_logger = TbLogger(tb_dir)

    # Print and save the run args
    pp.pprint(vars(opts))
    save_opts = open(os.path.join(opts.save_dir, 'log_dir', 'options_{}.txt'.format(opts.time_txt)), 'w')
    for k, v in vars(opts).items():
        save_opts.write("'{}': {}\n".format(k, v))
        print("'{}': {}".format(k, v))
    print()
    save_opts.close()

    # Set device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load problem
    problem = load_problem(opts.problem)

    # Load data
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Load model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork,
        'gpn': GPN,
        'gamma': GAMMA
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        num_depots=opts.num_depots,
        num_agents=opts.num_agents
    ).to(opts.device)

    # Multi-GPU
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Load baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()
    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Load optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Load learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Load validation data
    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset,
                                       distribution=opts.data_distribution, num_agents=opts.num_agents,
                                       num_depots=opts.num_depots, max_length=opts.max_length)

    # Resume training
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

        # Set the random states. Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # Evaluate validation data
    if opts.eval_only:
        validate(model, val_dataset, opts)

    # Train
    else:
        for epoch in range(opts.initial_epoch, opts.initial_epoch + opts.epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    run(get_options())
