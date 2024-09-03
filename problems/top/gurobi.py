import numpy as np
from gurobipy import Model, GRB, quicksum

def top_gurobi(num_agents, nodes, prizes, depot, max_length, timeout=0):
    num_nodes = len(nodes)
    nodes = np.array(nodes)
    depot = np.array(depot)
    
    # Calculate distance matrix
    all_nodes = np.vstack([depot, nodes])
    dist_matrix = np.linalg.norm(all_nodes[:, None] - all_nodes, axis=2)

    # Create Gurobi model
    model = Model('Team Orienteering Problem')
    model.Params.outputFlag = False
    
    # Decision variables
    x = model.addVars(num_agents, num_nodes + 1, num_nodes + 1, vtype=GRB.BINARY, name='x')
    u = model.addVars(num_agents, num_nodes + 1, vtype=GRB.CONTINUOUS, name='u')
    
    # Objective function: maximize total prize collected
    model.setObjective(
        quicksum(prizes[j-1] * x[i, j, k] for i in range(num_agents) for j in range(1, num_nodes + 1) for k in range(num_nodes + 1)),
        GRB.MAXIMIZE
    )
    
    # Constraints
    for i in range(num_agents):
        
        # Start from the depot
        model.addConstr(quicksum(x[i, 0, j] for j in range(1, num_nodes + 1)) == 1)
        
        # End at the depot
        model.addConstr(quicksum(x[i, j, 0] for j in range(1, num_nodes + 1)) == 1)
        
        # Do not go from node j to node j
        model.addConstr(quicksum(x[i, j, j] for j in range(1, num_nodes + 1)) == 0)
        
        # Flow constraints
        for j in range(1, num_nodes + 1):
            model.addConstr(quicksum(x[i, k, j] for k in range(num_nodes + 1)) == quicksum(x[i, j, k] for k in range(num_nodes + 1)))
        
        # Sub-tour elimination constraints
        for j in range(1, num_nodes + 1):
            for k in range(1, num_nodes + 1):
                if j != k:
                    model.addConstr(u[i, j] - u[i, k] + num_nodes * x[i, j, k] <= num_nodes - 1)

        # Distance constraint
        model.addConstr(quicksum(dist_matrix[j, k] * x[i, j, k] for j in range(num_nodes + 1) for k in range(num_nodes + 1)) <= max_length[i])
    
    # Add constraint to ensure that each non-depot node is visited by at most one agent
    for j in range(1, num_nodes + 1):
        model.addConstr(quicksum(x[i, k, j] for i in range(num_agents) for k in range(num_nodes + 1)) <= 1)
    
    # Optimize the model
    if timeout:
        model.Params.timeLimit = timeout
    model.Params.lazyConstraints = 1
    model.Params.threads = 0
    model.optimize()
    
    # Get the results
    results = []
    for i in range(num_agents):
        path = []
        current = 0  # Start from depot
        visited = set()  # Track visited nodes
        while True:
            path.append(current)
            visited.add(current)
            
            # Find the next node
            next_node = [k for k in range(num_nodes + 1) if x[i, current, k].X > 0.5 and k not in visited]
            
            # No more unvisited nodes
            if not next_node:
                break
            current = next_node[0]
        results.append(path)
    
    return results, model.Runtime
