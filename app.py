import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import load_model
from visualize import baselines
from utils.data_utils import set_seed
from utils.functions import load_problem
from visualize import reshape_tours, add_depots, assign_colors
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork
from nets.gpn import GPN


def load_scenario(seed, problem, graph_size, data_distribution, num_agents, num_depots, max_length, device):
    
    # Set seed for reproducibility
    set_seed(seed)

    # Load problem and dataset
    problem_instance = load_problem(problem)
    dataset = problem_instance.make_dataset(
        size=graph_size, num_samples=1, distribution=data_distribution,
        max_length=max_length, num_agents=num_agents, num_depots=num_depots
    )
    
    # Get scenario
    inputs = dataset.data[0]
    return inputs, dataset


def apply_model(model, num_agents, num_depots, graph_size, inputs, dataset, return2depot, timeout, use_cuda):
    
    # Apply a baseline (GA, PSO, ACO)
    if model in ['aco', 'pso', 'opga', 'gurobi', 'compass']:
        tours, inputs, model_name = baselines(num_agents, model, dataset, return2depot=return2depot, timeout=timeout)

    # Apply a Deep Learning model (Transformer, PN, GPN)
    else:
        # Set the device
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Load model (Transformer, PN, GPN) for evaluation on the chosen device
        model, _ = load_model(model, num_agents=num_agents)
        model.set_decode_type('greedy')
        model.num_depots = num_depots
        model.num_agents = num_agents
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
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            inputs[k] = v.unsqueeze(0).to(device)
        _, _, tours = model(inputs, return_pi=True)

        # Reshape tours list
        print(tours)
        tours = tours.cpu().detach().numpy().squeeze()
        print(tours)
        tours = reshape_tours(tours, num_agents, end_ids=graph_size + 1)
        print(inputs['loc'].shape[0] + 1)

    # Add depots and print tours
    tours = add_depots(tours, num_agents, graph_size)
    
    return tours, inputs, model_name
    

def plot(tours, scenario, problem, model_name, data_dist):
    
    # Torch tensors to numpy
    for k, v in scenario.items():
        if isinstance(v, torch.Tensor):
            scenario[k] = v.cpu().detach().numpy()
        scenario[k] = scenario[k].squeeze()
    
    # Plot
    fig, ax = plot_tour_streamlit(tours, scenario, problem, model_name, data_dist)


def plot_tour_streamlit(tours, inputs, problem, model_name, data_dist=''):
    """
    Plot a given tour using Plotly for Streamlit.
    """
    # Number of agents
    num_agents = len(tours)
    colors = assign_colors(num_agents) if num_agents <= 6 else np.random.rand(num_agents, 3)

    # Depot and locations
    depot = inputs['depot']
    depot2 = inputs['depot2'] if 'depot2' in inputs else depot
    loc = inputs['loc']
    prizes = inputs['prize']

    # Create a Plotly figure
    fig = go.Figure()

    # Plot depot
    fig.add_trace(go.Scatter(x=[depot[0]], y=[depot[1]], mode='markers', marker=dict(color='red', size=10),
                             name='Depot'))
    if 'depot2' in inputs:
        fig.add_trace(go.Scatter(x=[depot2[0]], y=[depot2[1]], mode='markers', marker=dict(color='blue', size=10),
                                 name='Depot 2'))

    # Plot nodes
    fig.add_trace(go.Scatter(x=loc[:, 0], y=loc[:, 1], mode='markers', marker=dict(color='cyan', size=10),
                             name='Nodes'))
    loc = np.concatenate(([depot], loc, [depot2]), axis=0)

    # Prizes (add prize 0 to depots)
    if len(inputs['prize']) != len(loc):
        prizes = np.concatenate(([0], inputs['prize'], [0]), axis=0)

    # Add tours
    for k, tour in enumerate(tours):
        nodes = np.take(loc, tour, axis=0)
        x_coords = nodes[:, 0]
        y_coords = nodes[:, 1]

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines+markers',
            line=dict(color=f'rgb({colors[k][0] * 255}, {colors[k][1] * 255}, {colors[k][2] * 255})', width=2),
            name=f'Agent {k + 1}'
        ))

    # Set title
    title = f"{problem.upper()} {num_agents}"
    if data_dist:
        title += f" ({data_dist.lower()})"
    title += f" - {model_name}: Max length = ..."
    fig.update_layout(title=title, xaxis_title="X-axis", yaxis_title="Y-axis", showlegend=True)

    # Render the plot in Streamlit
    st.plotly_chart(fig)
    return fig, None


def main():
    st.title("Team Orienteering Problem")

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    seed = st.sidebar.number_input("Random Seed", value=0)
    model = st.sidebar.text_input("Model Path or Baseline", value="opga")
    timeout = 0  # st.sidebar.number_input("Timeout (seconds)", value=0)
    problem = "top"
    graph_size = st.sidebar.number_input("Graph Size", value=20)
    data_distribution = st.sidebar.text_input("Data Distribution", value="const")
    num_agents = st.sidebar.number_input("Number of Agents", value=2)
    num_depots = 1
    return2depot = True
    max_length = 2.0  # st.sidebar.number_input("Max Length", value=2.0)
    use_cuda = st.sidebar.checkbox("Use CUDA", value=True)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load the scenario and dataset
    scenario, dataset = load_scenario(seed, problem, graph_size, data_distribution, num_agents, num_depots, max_length, device)

    # Always plot the scenario
    plot([], scenario, problem, "Scenario", data_distribution)

    # Buttons for calculating and cleaning
    col1, col2 = st.columns(2)
    with col1:
        calculate = st.button("Calculate")
    with col2:
        clean = st.button("Clean")

    # Calculate button action
    if calculate:
        tours, scenario, model_name = apply_model(model, num_agents, num_depots, graph_size, scenario, dataset, return2depot, timeout, use_cuda)
        
        # Plot the scenario with routes
        plot(tours, scenario, problem, model_name, data_distribution)

    # Clean button action
    if clean:
        pass

if __name__ == "__main__":
    main()
