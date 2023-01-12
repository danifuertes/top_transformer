import os
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt

from utils.data_utils import set_seed, str2bool, assign_colors
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork
from utils.functions import load_problem
from utils import load_model
from nets.gpn import GPN


def arguments(args=None):
    parser = argparse.ArgumentParser(description="Visualize predictions made by some algorithms")
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')

    # Method
    parser.add_argument('--model', help='Path to load model. Just indicate the directory where epochs are saved or'
                                        'the directory + the specific epoch you want to load. For baselines, indicate'
                                        'the name of the baselines instead (opga, pso, aco)-')

    # Problem
    parser.add_argument('--problem', default='top', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--data_distribution', type=str, default='const',
                        help='Data distribution to use during training, defaults and options depend on problem')
    parser.add_argument('--num_agents', type=int, default=4, help="Number of agents")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")
    parser.add_argument('--return2depot', type=str2bool, default=True, help="True for constraint of returning to depot")
    parser.add_argument('--max_length', type=float, default=2, help="Normalized time limit to solve the problem")

    # CPU / GPU
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Check problem is correct
    assert opts.problem == 'top', "Only the top problem is supported"
    assert opts.num_agents > 0, 'num_agents must be greater than 0'

    # Check baseline is correct for the given problem
    assert opts.model in ('opga', 'aco', 'pso') or os.path.exists(opts.model), \
        'Path to model does not exist. For baselines, the supported baselines for TOP are opga, aco, pso'
    return opts


# def force_return(tour, loc, max_length):
#     new_tour, distance = [tour[0]], 0
#     for i in range(len(tour) - 1):
#         distance += np.linalg.norm(loc[tour[i]] - loc[tour[i + 1]])
#         if distance + np.linalg.norm(loc[tour[-1]] - loc[tour[i + 1]]) > max_length:
#             new_tour.append(tour[-1])
#             break
#         new_tour.append(tour[i + 1])
#     return np.array(new_tour)


def baselines(num_agents, baseline, dataset, return2depot=True):

    # https://github.com/robin-shaun/Multi-UAV-Task-Assignment-Benchmark
    # https://github.com/dietmarwo/Multi-UAV-Task-Assignment-Benchmark

    # Prepare inputs
    inputs = dataset.data[0]
    for k, v in inputs.items():
        inputs[k] = v.detach().numpy()
    data = np.concatenate((inputs['loc'], np.expand_dims(inputs['prize'], 1), np.zeros((len(inputs['loc']), 1))), 1)
    data = np.concatenate((np.array([[*inputs['depot'], 0, 0]]), data), 0)

    # Genetic Algorithm
    if baseline == 'opga':
        from problems.top.opga import GA
        model_name = 'GA'
        tours, _ = GA(
            num_agents,
            np.array([1 for _ in range(num_agents)]),
            len(inputs['loc']),
            data,
            inputs['max_length'],
            return2depot
        ).run()

    # Particle Swarm Optimization
    if baseline == 'pso':
        from problems.top.pso import PSO
        model_name = 'PSO'
        tours, _ = PSO(
            num_agents,
            len(inputs['loc']),
            data,
            np.array([1 for _ in range(num_agents)]),
            inputs['max_length'],
            return2depot
        ).run()

    # Ant Colony Optimization
    if baseline == 'aco':
        from problems.top.aco import ACO
        model_name = 'ACO'
        tours, _ = ACO(
            num_agents,
            len(inputs['loc']),
            np.array([1 for _ in range(num_agents)]),
            data,
            inputs['max_length'],
            return2depot
        ).run()

    # Lists to numpy arrays
    for k, v in inputs.items():
        inputs[k] = np.array(v)

    return np.array(tours).squeeze(), inputs, model_name


def plot_tour(tours, inputs, problem, model_name, data_dist=''):
    """
    Plot a given tour.
    # Arguments
        tours (numpy array): contains one ordered list of nodes per agent.
        inputs (dict or numpy array): if TSP, inputs is an array containing the coordinates of the nodes. Otherwise, it
        is a dict with the coordinates of the nodes (loc) and the depot (depot), and other possible features.
        problem (str): name of the problem.
        model_name (str): name of the model.
        data_dist (str): type of prizes for the OP. For any other problem, just set this to ''.
    """

    # Number of agents
    num_agents = len(tours)
    colors = assign_colors(num_agents) if num_agents <= 6 else np.random.rand(num_agents, 3)
    # colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:pink', 'tab:purple']

    # Initialize plot
    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])

    # Depot (blue circle)
    depot = inputs['depot']
    plt.scatter(depot[0], depot[1], c='b')
    depot2 = inputs['depot2'] if 'depot2' in inputs else depot
    plt.scatter(depot2[0], depot2[1], c='r')  # (red circle)

    # Nodes (black circles)
    loc = inputs['loc']
    plt.scatter(loc[..., 0], loc[..., 1], c='k')
    loc = np.concatenate(([depot], loc, [depot2]), axis=0)

    # Prizes (add prize 0 to depots)
    if len(inputs['prize']) == len(loc):
        prizes = inputs['prize']
    else:
        prizes = np.concatenate(([0], inputs['prize'], [0]), axis=0)

    # For each agent
    reward, length = 0, 0
    for k, tour in enumerate(tours):

        # Calculate the length of the tour
        nodes = np.take(loc, tour, axis=0)
        d = np.sum(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1))
        length = length if length >= d else d
        reward += np.sum(np.take(prizes, tour))

        # Draw arrows
        for i in range(1, tour.shape[0]):
            dx = loc[tour[i], 0] - loc[tour[i - 1], 0]
            dy = loc[tour[i], 1] - loc[tour[i - 1], 1]
            plt.arrow(loc[tour[i - 1], 0], loc[tour[i - 1], 1], dx, dy, head_width=.025, fc=colors[k], ec=None,
                      length_includes_head=True)

    # Set title
    # title = 'Agents = {} |'.format(num_agents)
    # title += ' Max length = {:.3g}'.format(length)
    title = problem.upper()
    title += ' ' + str(num_agents) + ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else ''
    title += ' - {:s}: Max length = {:.3g}'.format(model_name, length)
    if problem == 'top':
        # Add TOP prize to the title (if problem is TOP)
        title += ' / {:.3g} | Prize = {:.3g} / {:.3g}'.format(inputs['max_length'], reward, np.sum(prizes))
    ax.set_title(title)
    plt.show()


def reshape_tours(tours, num_agents, end_ids=0):
    new_tours = [[] for _ in range(num_agents)]
    count, check = 0, True
    for node in tours.reshape(-1, order='F'):
        if count >= num_agents:
            break
        if node == end_ids:
            if check:
                count += 1
            check = False
        else:
            new_tours[count].append(node)
            check = True
    return new_tours


def add_depots(tours, num_agents, graph_size):
    tours = list(tours)
    for k in range(num_agents):
        tours[k] = np.array(tours[k])
        if len(tours[k]) > 0:
            if tours[k][0] != 0:
                tours[k] = np.concatenate(([0], tours[k]), axis=0)
            if tours[k][-1] != graph_size + 1:
                tours[k] = np.concatenate((tours[k], [graph_size + 1]), axis=0)
        else:
            tours[k] = np.array([0, graph_size + 1])
        print('Agent {}: '.format(k + 1), tours[k])
    return tours


def main(opts):

    # Set seed for reproducibility
    set_seed(opts.seed)

    # Load problem
    problem = load_problem(opts.problem)
    dataset = problem.make_dataset(size=opts.graph_size, num_samples=1, distribution=opts.data_distribution,
                                   max_length=opts.max_length, num_agents=opts.num_agents, num_depots=opts.num_depots)
    inputs = dataset.data[0]

    # Apply a baseline (GA, PSO, ACO)
    if opts.model in ['aco', 'pso', 'opga']:
        tours, inputs, model_name = baselines(opts.num_agents, opts.model, dataset, return2depot=opts.return2depot)

    # Apply a Deep Learning model (Transformer, PN, GPN)
    else:
        # Set the device
        device = torch.device("cuda:0" if opts.use_cuda else "cpu")

        # Load model (Transformer, PN, GPN) for evaluation on the chosen device
        model, _ = load_model(opts.model, num_agents=opts.num_agents)
        model.set_decode_type('greedy')
        model.num_depots = opts.num_depots
        model.num_agents = opts.num_agents
        model.eval()  # Put in evaluation mode to not track gradients
        model.to(device)
        if isinstance(model, AttentionModel):
            model_name = 'Transformer'
        elif isinstance(model, PointerNetwork):
            model_name = 'Pointer'
        else:
            assert isinstance(model, GPN), 'Model should be an instance of AttentionModel, PointerNetwork or GPN'
            model_name = 'GPN'

        # Calculate tour
        for k, v in inputs.items():
            inputs[k] = v.unsqueeze(0).to(device)
        _, _, tours = model(inputs, return_pi=True)

        # Torch tensors to numpy
        tours = tours.cpu().detach().numpy().squeeze()
        for k, v in inputs.items():
            inputs[k] = v.cpu().detach().numpy().squeeze()

        # Reshape tours list
        tours = reshape_tours(tours, opts.num_agents, end_ids=inputs['loc'].shape[0] + 1)

    # Add depots and print tours
    tours = add_depots(tours, opts.num_agents, opts.graph_size)

    # Plot tours
    plot_tour(tours, inputs, problem.NAME, model_name, data_dist=opts.data_distribution)


if __name__ == "__main__":
    main(arguments())
