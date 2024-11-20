import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set paths for the datasets
DATASETS = {
    "Credit Card Customer Data": '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/Credit Card Customer Data.csv',
    "Sport Car Price Data": '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/Sport car price.csv',
    "Student Data": '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/student_data.csv'
}

# Function to load dataset
def load_data(dataset_name):
    path = DATASETS[dataset_name]
    return pd.read_csv(path)

# Function to process dataset and return x and y
def process_data(dataset_name):
    df = load_data(dataset_name)
    df.columns = df.columns.str.lower()  # Standardize column names
    if dataset_name == "Credit Card Customer Data":
        x = df[['avg_credit_limit', 'total_credit_cards']].values
        y = df['avg_credit_limit'].values
    elif dataset_name == "Sport Car Price Data":
        x = df[['mileage', 'engine_size']].values  # Example features
        y = df['price'].values
    elif dataset_name == "Student Data":
        if 'age' not in df.columns:
            raise ValueError("The 'Age' column is missing in the dataset.")
        df.fillna({'age': df['age'].mean()}, inplace=True)
        x = df[['age', 'marks']].values  # Example features
        y = df['marks'].values
    else:
        raise ValueError("Unsupported dataset selected.")
    return x, y

# Streamlit setup
st.title("Three Algorithm Visualization")
st.sidebar.header("Configuration")

# Select dataset
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    list(DATASETS.keys()),
    key="dataset_selectbox"
)

# Select algorithm
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ("Select Algorithm", "Linear Regression", "K-Nearest Neighbour", "Support Vector Machine"),
    key="algorithm_selectbox"
)

# Graph setup
fig, ax = plt.subplots()

if dataset_name != "Select Dataset" and algorithm != "Select Algorithm":
    # Load and process data
    try:
        x, y = process_data(dataset_name)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if algorithm == "Linear Regression":
            if st.sidebar.button("Run Linear Regression"):
                model = LinearRegression()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                st.subheader(f"Mean Squared Error for Linear Regression: {round(mean_squared_error(y_test, y_pred), 2)}")
                ax.scatter(x[:, 0], y, color='blue', label="Data")
                ax.plot(x_test[:, 0], y_pred, color='red', label="Linear Fit")
                ax.legend()
                st.pyplot(fig)

        elif algorithm == "K-Nearest Neighbour":
            n_neighbors = st.sidebar.number_input("Enter number of neighbors", min_value=1, value=5, key="knn_neighbors")
            if st.sidebar.button("Run KNN Algorithm"):
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(x_train, y_train)
                y_pred = knn.predict(x_test)

                st.subheader(f"Accuracy for K-Nearest Neighbour: {round(accuracy_score(y_test, y_pred), 2)}")
                ax.scatter(x[:, 0], y, color='green', label="Data")
                st.pyplot(fig)

        elif algorithm == "Support Vector Machine":
            C = st.sidebar.number_input("Enter C", min_value=0.0, value=1.0, key="svm_c")
            kernel = st.sidebar.selectbox('Enter Kernel', ('rbf', 'linear', 'poly', 'sigmoid'), key="svm_kernel")
            gamma = st.sidebar.selectbox('Enter Gamma', ('scale', 'auto'), key="svm_gamma")

            if st.sidebar.button("Run SVM Algorithm"):
                svm = SVC(C=C, kernel=kernel, gamma=gamma)
                svm.fit(x_train, y_train)
                y_pred = svm.predict(x_test)

                st.subheader(f"Accuracy for Support Vector Machine: {round(accuracy_score(y_test, y_pred), 2)}")
                ax.scatter(x[:, 0], y, color='orange', label="Data")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing dataset: {e}")

st.pyplot(fig)
