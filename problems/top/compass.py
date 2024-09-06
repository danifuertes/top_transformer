import time
import numpy as np

def calculate_distance(node1, node2):
    """Calculate Euclidean distance between two nodes."""
    return np.linalg.norm(np.array(node1) - np.array(node2))

def top_compass(num_agents, nodes, prizes, depot, max_length):
    """
    Solve the team orienteering problem using a compass heuristic approach.
    
    Parameters:
    - num_agents: Number of agents to solve the team orienteering problem.
    - nodes: List of tuples representing nodes with normalized coordinates.
    - prizes: List of prizes for visiting each node.
    - depot: Tuple representing the depot node with normalized coordinates.
    - max_length: List indicating the maximum distance allowed for each agent.
    
    Returns:
    - List of lists containing the sequence of node indices visited by each agent.
    """
    duration = time.time()
    
    # Initialize
    num_nodes = len(nodes)
    visited = [False] * num_nodes
    paths = [[] for _ in range(num_agents)]
    total_distances = [0] * num_agents

    # Iterate over each agent
    for agent in range(num_agents):
        current_node = depot
        path = []
        available_distance = max_length[agent]

        while True:
            best_ratio = 0
            best_node = None
            best_node_index = None

            for i in range(num_nodes):
                if not visited[i]:
                    distance_to_node = calculate_distance(current_node, nodes[i])
                    distance_to_depot = calculate_distance(nodes[i], depot)
                    
                    # Ensure that the agent can return to the depot after visiting the node
                    if distance_to_node + distance_to_depot <= available_distance:
                        ratio = prizes[i] / distance_to_node if distance_to_node > 0 else float('inf')
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_node = nodes[i]
                            best_node_index = i

            if best_node is None:
                # No more nodes can be visited within the available distance
                break

            # Visit the selected best node
            path.append(best_node_index)
            visited[best_node_index] = True
            available_distance -= calculate_distance(current_node, best_node)
            total_distances[agent] += calculate_distance(current_node, best_node)
            current_node = best_node

        # Return to the depot
        total_distances[agent] += calculate_distance(current_node, depot)
        if total_distances[agent] <= max_length[agent]:
            paths[agent] = path

    return [[p+1 for p in path] for path in paths], time.time() - duration