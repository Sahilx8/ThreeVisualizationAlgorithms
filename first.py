import streamlit as st
import numpy as np
import pandas as pd


# KNN Helper Function
def knn(data, features, input_features, k):
    distances = np.sqrt(((data[features] - input_features) ** 2).sum(axis=1))
    data['Distance'] = distances
    return data.nsmallest(k, 'Distance')


# Sidebar Selection
st.sidebar.markdown("# K-Nearest Neighbors")
Dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Select Dataset', 'Credit Card Customer Data', 'Sport Car Price', 'Student Data')
)

# Dataset Paths
datasets_path = "/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/"

if Dataset == 'Credit Card Customer Data':
    file_path = datasets_path + "Credit Card Customer Data.csv"
    try:
        cc_data = pd.read_csv(file_path)
        st.write("Dataset Preview:", cc_data.head())

        # Input Features
        avg_credit_limit = st.sidebar.number_input("Enter Avg Credit Limit", min_value=0)
        current_balance = st.sidebar.number_input("Enter Current Balance", min_value=0)
        k = st.sidebar.number_input("Enter k", min_value=1)

        # Run KNN
        if st.sidebar.button("Run Algorithm"):
            input_features = np.array([avg_credit_limit, current_balance])
            knn_result = knn(cc_data, ['Avg_credit_card_limit', 'Current Balance'], input_features, k)
            st.subheader("K Nearest Customers")
            st.table(knn_result.drop(columns=['Distance']))
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")

elif Dataset == 'Sport Car Price':
    file_path = datasets_path + "Sport car price.csv"
    try:
        car_data = pd.read_csv(file_path)
        if car_data['Price'].isnull().all():  # Generate synthetic price if needed
            car_data['Price'] = (
                (2025 - car_data['Prod. year'].astype(int)) * 1000 +
                car_data['Engine volume'].str.extract(r'(\d+.\d+)').astype(float).fillna(1) * 500 +
                np.random.randint(5000, 20000, size=len(car_data))
            )
        st.write("Dataset Preview:", car_data.head())

        # Input Features
        horsepower = st.sidebar.number_input("Enter Horsepower", min_value=0)
        weight = st.sidebar.number_input("Enter Weight", min_value=0)
        k = st.sidebar.number_input("Enter k", min_value=1)

        # Run KNN
        if st.sidebar.button("Run Algorithm"):
            input_features = np.array([horsepower, weight])
            knn_result = knn(car_data, ['Horsepower', 'Weight'], input_features, k)
            st.subheader("K Nearest Cars")
            st.table(knn_result.drop(columns=['Distance']))
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")

elif Dataset == 'Student Data':
    file_path = datasets_path + "student_data.csv"
    try:
        student_data = pd.read_csv(file_path)
        st.write("Dataset Preview:", student_data.head())

        # Input Features
        age = st.sidebar.number_input("Enter Age", min_value=0)
        gpa = st.sidebar.number_input("Enter GPA", min_value=0.0, max_value=4.0)
        k = st.sidebar.number_input("Enter k", min_value=1)

        # Run KNN
        if st.sidebar.button("Run Algorithm"):
            input_features = np.array([age, gpa])
            knn_result = knn(student_data, ['Age', 'GPA'], input_features, k)
            st.subheader("K Nearest Students")
            st.table(knn_result.drop(columns=['Distance']))
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")

else:
    st.info("Select a dataset to begin.")
