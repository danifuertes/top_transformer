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
from nets.gamma import GAMMA
from nets.gpn import GPN


def load_scenario(seed, problem, graph_size, data_distribution, num_agents, num_depots, max_length):
    
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
        elif isinstance(model, GAMMA):
            model_name = 'GAMMA'
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
        tours = tours.cpu().detach().numpy().squeeze()
        tours = reshape_tours(tours, num_agents, end_ids=graph_size + 1)

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
    plot_tour_streamlit(tours, scenario, problem, model_name, data_dist)


def plot_tour_streamlit(tours, inputs, problem, model_name, data_dist=''):
    """
    Plot a given tour using Plotly for Streamlit.
    """
    # Number of agents
    num_agents = len(tours) if len(tours) > 0 else 1
    colors = assign_colors(num_agents) if num_agents <= 6 else np.random.rand(num_agents, 3)

    # Depot and locations
    depot = inputs['depot']
    depot2 = inputs['depot2'] if 'depot2' in inputs else depot
    loc = inputs['loc']
    prizes = inputs['prize']
    max_length = inputs['max_length']

    # Create a Plotly figure
    fig = go.Figure()

    # Plot depot
    fig.add_trace(
        go.Scatter(x=[depot[0]], y=[depot[1]], mode='markers', name='Depot', marker=dict(
            color='deepskyblue',
            size=25,
            symbol='triangle-up',
            line=dict(color='black', width=1)
        ))
    )
    if 'depot2' in inputs:
        fig.add_trace(
            go.Scatter(x=[depot2[0]], y=[depot2[1]], mode='markers', name='Depot 2', marker=dict(
                color='limegreen',
                size=25,
                symbol='triangle-up',
                line=dict(color='black', width=1)
            ))
        )

    # Plot nodes
    marker_opacity = 0.5 * (1 + prizes)
    marker_size = 5 * (2 + 3*prizes)
    fig.add_trace(
        go.Scatter(x=loc[:, 0], y=loc[:, 1], mode='markers', name='Nodes', marker=dict(
            color='red',
            size=marker_size,
            opacity=marker_opacity,
            line=dict(color='black', width=1)
        ))
    )
    loc = np.concatenate(([depot], loc, [depot2]), axis=0)

    # Prizes (add prize 0 to depots)
    if len(inputs['prize']) != len(loc):
        prizes = np.concatenate(([0], inputs['prize'], [0]), axis=0)

    # Add tours
    num_nodes, reward, length = 0, 0, 0
    for k, tour in enumerate(tours):
        nodes = np.take(loc, tour, axis=0)
        num_nodes += len(nodes) - 2
        x_coords = nodes[:, 0]
        y_coords = nodes[:, 1]

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines+markers',
            line=dict(color=f'rgb({colors[k][0] * 255}, {colors[k][1] * 255}, {colors[k][2] * 255})', width=2),
            name=f'Agent {k + 1}'
        ))
        
        reward += prizes[tour].sum()
        length += np.sum(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1))

    # Set title
    title = f"Number of Nodes: {num_nodes} / {len(loc) - 2}        \
              Path length: {length:.2f} / {num_agents * max_length:.0f}        \
              Reward collected: {reward:.2f} / {prizes.sum():.2f}"
    fig.update_layout(
        title=title,
        showlegend=True,
        autosize=False,
        width=700,
        height=700,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, 1],
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, 1],
        ),
    )
    fig.update_xaxes(constrain='domain')  
    fig.update_yaxes(scaleanchor= 'x')
    
    # Add four lines to form the square borders
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],  # Left border
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],  # Bottom border
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[1, 1], y=[0, 1],  # Right border
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 1],  # Top border
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))

    # Render the plot in Streamlit
    st.plotly_chart(fig)
    return fig, None


def main():
    # Set name and icon to webpage
    st.set_page_config(
        page_title="TOP Demo",
        page_icon="🧭",
    )
    st.title("Team Orienteering Problem")

    # Initialize session state for tours
    st.session_state.tours = []
    model_name = "Scenario"

    # Sidebar for input parameters
    st.sidebar.header("Inputs")
    seed = st.sidebar.number_input("Random Seed", value=0)
    model = st.sidebar.selectbox("Model", ["Transformer", "Pointer Network", "Graph Pointer Network", "GAMMA", "Genetic Algorithm", "Compass", "ACO", "PSO", "Gurobi"], index=0)
    timeout = 0  # st.sidebar.number_input("Timeout (seconds)", value=0)
    problem = "top"
    graph_size = st.sidebar.number_input("Graph Size", value=20, min_value=20, max_value=100)
    data_distribution = st.sidebar.selectbox("Reward Distribution", ["Constant", "Uniform", "Distance to depot"], index=0)
    # num_agents = st.sidebar.number_input("Number of Agents", value=2, min_value=2, max_value=5)
    num_depots = 1
    return2depot = True
    max_length = 2.0  # st.sidebar.number_input("Max Length", value=2.0)
    use_cuda = st.sidebar.checkbox("Use CUDA", value=True)
    
    # Map inputs to real values
    data_distribution = {"Constant": "const", "Uniform": "unif", "Distance to depot": "dist"}[data_distribution]
    model = {"Transformer": "transformer", "Pointer Network": "pointer", "Graph Pointer Network": "gpn", "GAMMA": "gamma", "Genetic Algorithm": "opga", "Compass": "compass", "ACO": "aco", "PSO": "pso", "Gurobi": "gurobi"}[model]
    model_graph_size = 20 if graph_size <= 30 else (50 if graph_size <= 75 else 100)
    num_agents = 2 if model_graph_size == 20 else (3 if model_graph_size == 50 else 5)
    if model in ('transformer', 'pointer', 'gpn', 'gamma'):
        if model == 'transformer':
            model_model = 'attention'
        elif model == 'pn':
            model_model = 'pointer'
        else:
            model_model = model
        model = f"pretrained/{problem}_{data_distribution}{model_graph_size}/{model_model}_rollout_{num_agents}agents_L{max_length}"

    # Load the scenario and dataset
    scenario, dataset = load_scenario(seed, problem, graph_size, data_distribution, num_agents, num_depots, max_length)

    # Buttons for calculating and cleaning
    col1, col2 = st.sidebar.columns(2)
    with col1:
        calculate = st.sidebar.button("Calculate", type='primary')
    with col2:
        clean = st.sidebar.button("Clean")

    # Calculate button action
    if calculate:
        tours, scenario, model_name = apply_model(model, num_agents, num_depots, graph_size, scenario, dataset, return2depot, timeout, use_cuda)
        st.session_state.tours = tours

    # Clean button action
    if clean:
        st.session_state.tours = []
        
    # Plot the scenario with routes
    plot(st.session_state.tours, scenario, problem, model_name, data_distribution)

if __name__ == "__main__":
    main()
