import streamlit as st
import matplotlib.pyplot as plt
from visualize import arguments, main

def main_app():
    st.title("Visualization of Predictions")

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    seed = st.sidebar.number_input("Random Seed", value=0)
    model = st.sidebar.text_input("Model Path or Baseline", value="opga")
    timeout = st.sidebar.number_input("Timeout (seconds)", value=0)
    problem = st.sidebar.text_input("Problem", value="top")
    graph_size = st.sidebar.number_input("Graph Size", value=20)
    data_distribution = st.sidebar.text_input("Data Distribution", value="const")
    num_agents = st.sidebar.number_input("Number of Agents", value=2)
    num_depots = st.sidebar.number_input("Number of Depots", value=1)
    return2depot = st.sidebar.checkbox("Return to Depot", value=True)
    max_length = st.sidebar.number_input("Max Length", value=2.0)
    use_cuda = st.sidebar.checkbox("Use CUDA", value=True)

    # Run the visualization when the button is clicked
    if st.sidebar.button("Run Visualization"):
        args = [
            '--seed', str(seed),
            '--model', model,
            '--timeout', str(timeout),
            '--problem', problem,
            '--graph_size', str(graph_size),
            '--data_distribution', data_distribution,
            '--num_agents', str(num_agents),
            '--num_depots', str(num_depots),
            '--return2depot', str(return2depot),
            '--max_length', str(max_length),
            '--use_cuda', str(use_cuda)
        ]
        opts = arguments(args)
        main(opts)

        # Display the plot
        st.pyplot(plt)

if __name__ == "__main__":
    main_app()
