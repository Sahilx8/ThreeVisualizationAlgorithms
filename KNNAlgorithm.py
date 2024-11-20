import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.sidebar.markdown("# K-Nearest Neighbors")

# Select dataset from the sidebar
Dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Select Dataset', 'Credit Card Customer Data', 'Sport Car Price Data', 'Student Data')
)

# Set the path to your datasets
data_path = '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/'

def calculate_distances(data, input_features, metric='Euclidean'):
    """Calculate distances using the selected metric."""
    if metric == 'Euclidean':
        return np.linalg.norm(data.values - np.array(input_features), axis=1)
    elif metric == 'Manhattan':
        return np.abs(data.values - np.array(input_features)).sum(axis=1)
    else:
        raise ValueError("Unsupported distance metric.")

def knn_algorithm(data, input_features, k, metric='Euclidean'):
    """Perform KNN and return the nearest neighbors."""
    distances = calculate_distances(data, input_features, metric)
    nearest_neighbors = []
    for _ in range(k):
        min_index = np.argmin(distances)
        nearest_neighbors.append(data.iloc[min_index])
        distances[min_index] = float('inf')  # Exclude the closest neighbor for the next iteration
    return nearest_neighbors

if Dataset != 'Select Dataset':
    # Load the selected dataset
    dataset_file_map = {
        'Credit Card Customer Data': "Credit Card Customer Data.csv",
        'Sport Car Price Data': "Sport car price.csv",
        'Student Data': "student_data.csv"
    }
    dataset = pd.read_csv(data_path + dataset_file_map[Dataset])
    st.write(dataset.head())

    # Feature selection
    available_features = list(dataset.columns)
    selected_features = st.sidebar.multiselect("Select Features", available_features)

    if selected_features:
        # Collect user inputs for each selected feature
        user_input = [
            st.sidebar.number_input(f"Enter {feature}", value=0.0) for feature in selected_features
        ]
        k = st.sidebar.number_input("Enter k", min_value=1, max_value=len(dataset), value=3)

        # Select distance metric
        distance_metric = st.sidebar.selectbox(
            "Select Distance Metric", ['Euclidean', 'Manhattan'], index=0
        )

        if st.sidebar.button("Run Algorithm"):
            if k > len(dataset):
                st.error("k cannot be greater than the dataset size.")
            else:
                nearest_neighbors = knn_algorithm(
                    dataset[selected_features], user_input, k, metric=distance_metric
                )
                
                # Display results
                st.subheader("K Nearest Neighbors")
                st.dataframe(pd.DataFrame(nearest_neighbors))

                # Visualization (only for 2D features)
                if len(selected_features) == 2:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(dataset[selected_features[0]], dataset[selected_features[1]], label="Dataset")
                    plt.scatter(
                        [user_input[0]], [user_input[1]], color='red', label="Input Point", marker="x"
                    )
                    plt.title("KNN Visualization")
                    plt.xlabel(selected_features[0])
                    plt.ylabel(selected_features[1])
                    plt.legend()
                    st.pyplot(plt)
    else:
        st.warning("Please select at least one feature to proceed.")
else:
    st.write("Select a dataset from the sidebar to begin.")
