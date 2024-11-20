import streamlit as st
import pandas as pd  # Ensure pandas is imported
from KNNAlgorithm import knn_algorithm  # Import the KNN function
from ThreeAlgorithmVisualization import run_three_algorithms  # Import the visualization function

def main():
    st.title("Machine Learning Algorithms")

    # Navigation Sidebar
    option = st.sidebar.selectbox(
        "Select Algorithm",
        ("Select Algorithm", "K-Nearest Neighbors", "Three Algorithms Visualization")
    )

    if option == "K-Nearest Neighbors":
        st.sidebar.header("K-Nearest Neighbors")
        run_knn_algorithm()  # Call the function to run the KNN algorithm

    elif option == "Three Algorithms Visualization":
        st.sidebar.header("Three Algorithms Visualization")
        run_three_algorithms()  # Call the function for visualization

def run_knn_algorithm():
    st.sidebar.markdown("# Configure KNN")
    
    # Path to the datasets
    data_path = '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/'

    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Select Dataset",
        ("Select Dataset", "Credit Card Customer Data", "Sport car price Data", "Student Data")
    )

    if dataset != "Select Dataset":
        # Map dataset names to file paths
        dataset_file_map = {
            "Credit Card Customer Data": "Credit Card Customer Data.csv",
            "Sport car price Data": "Sport car price.csv",
            "Student Data": "student_data.csv"
        }
        # Load the selected dataset
        try:
            data = pd.read_csv(data_path + dataset_file_map[dataset])
            st.write(data.head())  # Display the first few rows of the dataset

            # Feature selection
            features = st.sidebar.multiselect("Select Features", data.columns)

            if features:
                user_input = [st.sidebar.number_input(f"Enter {feature}", value=0.0) for feature in features]
                k = st.sidebar.number_input("Enter k", min_value=1, max_value=len(data), value=3)
                distance_metric = st.sidebar.selectbox("Select Distance Metric", ["Euclidean", "Manhattan"], index=0)

                if st.sidebar.button("Run Algorithm"):
                    if k > len(data):
                        st.error("k cannot be greater than the dataset size.")
                    else:
                        # Perform KNN
                        nearest_neighbors = knn_algorithm(data[features], user_input, k, metric=distance_metric)
                        st.subheader("K Nearest Neighbors")
                        st.dataframe(pd.DataFrame(nearest_neighbors))

                        # Visualization (2D case)
                        if len(features) == 2:
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(8, 6))
                            plt.scatter(data[features[0]], data[features[1]], label="Dataset")
                            plt.scatter([user_input[0]], [user_input[1]], color="red", label="Input Point", marker="x")
                            plt.title("KNN Visualization")
                            plt.xlabel(features[0])
                            plt.ylabel(features[1])
                            plt.legend()
                            st.pyplot(plt)
            else:
                st.warning("Please select at least one feature to proceed.")
        except FileNotFoundError:
            st.error("File not found. Please check the file path.")
    else:
        st.write("Select a dataset from the sidebar to begin.")

if __name__ == "__main__":
    main()
