import os
import re
import argparse
import numpy as np
from datetime import timedelta

from problems.top.opga import GA
from problems.top.aco import ACO
from problems.top.pso import PSO
from utils import run_all_in_pool
#from visualize import force_return
from problems.top.gurobi import top_gurobi
from utils.data_utils import check_extension, load_dataset, save_dataset, str2bool, set_seed

MAX_LENGTH_TOL = 1e-5


def calc_op_total(prize, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    return np.array(prize)[np.array(tour) - 1].sum() if len(tour) > 0 else 0


def calc_op_length(depot, loc, tour, return2depot=True):
    if len(tour) > 0:
        assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
        loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
        tour_with_depot = ([0], tour, [0]) if return2depot else ([0], tour)
        sorted_locs = loc_with_depot[np.concatenate(tour_with_depot)]
        return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
    return 0


def solve_opga(directory, name, depot, loc, prize, max_length, num_agents, return2depot=True, disable_cache=False, *args, **kwargs):
    problem_filename = os.path.join(directory, "{}.opga.pkl".format(name))
    data = np.concatenate((loc, np.expand_dims(prize, 1), np.zeros((len(loc), 1))), 1)
    data = np.concatenate((np.array([[*depot, 0, 0]]), data), 0)
    if os.path.isfile(problem_filename) and not disable_cache:
        (cost, tours, duration) = load_dataset(problem_filename)
    else:
        tours, duration = GA(
            num_agents,
            np.array([1 for _ in range(num_agents)]),
            len(loc),
            data,
            max_length,
            return2depot
        ).run()
        # if return2depot:
        #     tours = [
        #         force_return(np.concatenate(([0], tour, [0])), np.concatenate(([depot], loc), 0), max_length)[1:-1]
        #         for tour in tours
        #     ]
        cost = np.sum([-calc_op_total(prize, tour) for tour in tours])
        save_dataset((cost, tours, duration), problem_filename)

    for tour in tours:
        assert calc_op_length(depot, loc, tour, return2depot) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
    return cost, tours, duration


def solve_aco(directory, name, depot, loc, prize, max_length, num_agents, return2depot=True, disable_cache=False, *args, **kwargs):
    problem_filename = os.path.join(directory, "{}.aco.pkl".format(name))
    data = np.concatenate((loc, np.expand_dims(prize, 1), np.zeros((len(loc), 1))), 1)
    data = np.concatenate((np.array([[*depot, 0, 0]]), data), 0)
    if os.path.isfile(problem_filename) and not disable_cache:
        (cost, tours, duration) = load_dataset(problem_filename)
    else:
        tours, duration = ACO(
            num_agents,
            len(loc),
            np.array([1 for _ in range(num_agents)]),
            data,
            max_length,
            return2depot
        ).run()
        # if return2depot:
        #     tours = [
        #         force_return(np.concatenate(([0], tour, [0])), np.concatenate(([depot], loc), 0), max_length)[1:-1]
        #         for tour in tours
        #     ]
        cost = np.sum([-calc_op_total(prize, tour) for tour in tours])
        save_dataset((cost, tours, duration), problem_filename)

    for tour in tours:
        assert calc_op_length(depot, loc, tour, return2depot) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
    return cost, tours, duration


def solve_pso(directory, name, depot, loc, prize, max_length, num_agents, return2depot=True, disable_cache=False, *args, **kwargs):
    problem_filename = os.path.join(directory, "{}.pso.pkl".format(name))
    data = np.concatenate((loc, np.expand_dims(prize, 1), np.zeros((len(loc), 1))), 1)
    data = np.concatenate((np.array([[*depot, 0, 0]]), data), 0)
    if os.path.isfile(problem_filename) and not disable_cache:
        (cost, tours, duration) = load_dataset(problem_filename)
    else:
        tours, duration = PSO(
            num_agents,
            len(loc),
            data,
            np.array([1 for _ in range(num_agents)]),
            max_length,
            return2depot
        ).run()
        # if return2depot:
        #     tours = [
        #         force_return(np.concatenate(([0], tour, [0])), np.concatenate(([depot], loc), 0), max_length)[1:-1]
        #         for tour in tours
        #     ]
        print(tours)
        cost = np.sum([-calc_op_total(prize, tour) for tour in tours])
        save_dataset((cost, tours, duration), problem_filename)

    for tour in tours:
        assert calc_op_length(depot, loc, tour, return2depot) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
    return cost, tours, duration


def solve_gurobi(directory, name, depot, loc, prize, max_length, num_agents, return2depot=True, disable_cache=False, timeout=0, *args, **kwargs):
    problem_filename = os.path.join(directory, "{}.gurobi.pkl".format(name))
    if os.path.isfile(problem_filename) and not disable_cache:
        (cost, tours, duration) = load_dataset(problem_filename)
    else:
        tours, duration = top_gurobi(
            num_agents=num_agents,
            nodes=loc,
            prizes=prize,
            depot=depot,
            max_length=np.ones(num_agents) * max_length,
            timeout=timeout,
        )
        tours = [tour[1:] for tour in tours]
        cost = np.sum([-calc_op_total(prize, tour) for tour in tours])
        save_dataset((cost, tours, duration), problem_filename)

    for tour in tours:
        assert calc_op_length(depot, loc, tour, return2depot) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
    return cost, tours, duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="Random seed to use")

    # Data
    parser.add_argument('--problem', default='top', help="The problem to solve")
    parser.add_argument('--num_agents', type=int, default=2, help="Number of agents")
    parser.add_argument('--return2depot', type=str2bool, default=True, help="True for constraint of returning to depot")
    parser.add_argument('--datasets', nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('-n', type=int, help="Number of instances to process")

    # Results
    parser.add_argument('-f', action='store_true', help="Set true to overwrite")
    parser.add_argument('-o', default=None, help="Name of the results file to write")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help="Minimum interval")
    parser.add_argument('--disable_cache', action='store_true', help="Disable caching")

    # Method
    parser.add_argument('--method', help="Name of the method to evaluate, 'aco', 'opga', 'pso', or 'gurobi'")
    parser.add_argument('--timeout', type=int, default=0, help="Number of seconds to retrieve gurobi solution")

    # CPU
    parser.add_argument('--multiprocessing', type=str2bool, default=False, help="Use multiprocessing")
    parser.add_argument('--cpus', type=int, help="Number of CPUs to use, defaults to all cores")

    opts = parser.parse_args()
    set_seed(opts.seed)
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:
        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"
        dataset_basename, ext = os.path.splitext(dataset_path.replace('/', '_'))

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, opts.problem, dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.path.join(results_dir, "{}{}".format(opts.method, ext))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        target_dir = os.path.join(results_dir, opts.method)
        assert opts.f or not os.path.isdir(target_dir), \
            "Target dir already exists! Try running with -f option to overwrite."

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        dataset = load_dataset(dataset_path)

        if method == "opga":
            def run_func(args):
                return solve_opga(*args, num_agents=opts.num_agents, return2depot=opts.return2depot,
                                  disable_cache=opts.disable_cache)

        elif method == "aco":
            def run_func(args):
                return solve_aco(*args, num_agents=opts.num_agents, return2depot=opts.return2depot,
                                 disable_cache=opts.disable_cache)

        elif method == "pso":
            def run_func(args):
                return solve_pso(*args, num_agents=opts.num_agents, return2depot=opts.return2depot,
                                 disable_cache=opts.disable_cache)

        else:
            assert method == "gurobi"
            def run_func(args):
                return solve_gurobi(*args, num_agents=opts.num_agents, return2depot=opts.return2depot, timeout=opts.timeout,
                                    disable_cache=opts.disable_cache)

        results, parallelism = run_all_in_pool(
            run_func,
            target_dir, dataset, opts, use_multiprocessing=opts.multiprocessing
        )

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print('Min cost: {} | Max cost: {}'.format(np.min(costs), np.max(costs)))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {} | ({:.4f} seconds)"
              .format(timedelta(seconds=int(np.sum(durations) / parallelism)), np.sum(durations) / parallelism))
        nodes = []
        for tour in tours:
            nodes.append(0)
            for k in range(len(tour)):
                nodes[-1] += len(tour[k])
        print('Average number of nodes visited: {} +- {}'
              .format(np.mean(nodes), 2 * np.std(nodes) / np.sqrt(len(nodes))))
        save_dataset((results, parallelism), out_file)
