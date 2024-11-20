import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set the paths for the datasets
DATA_PATH = '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/'

def load_data():
    """Loads and preprocesses the datasets."""
    # Load datasets
    cc_data = pd.read_csv(DATA_PATH + "Credit Card Customer Data.csv")
    car_data = pd.read_csv(DATA_PATH + "Sport car price.csv")
    student_data = pd.read_csv(DATA_PATH + "student_data.csv")

    # Preprocess Sport Car dataset to handle missing 'Price (in USD)' column
    car_data['Price'] = pd.to_numeric(car_data['Price'], errors='coerce')
    car_data.fillna(car_data.median(), inplace=True)  # Fill missing price values with the median

    return cc_data, car_data, student_data

def run_algorithm(X, y, algorithm, **kwargs):
    """Runs the selected predictive algorithm and returns results."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = algorithm(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def display_results(dataset_name, data, feature_col, target_col, algorithm_name, algorithm, **kwargs):
    """Displays results of the selected algorithm."""
    st.subheader(f"{dataset_name} Data")
    st.write(data.head())

    X = data[[feature_col]]
    y = data[target_col]

    if st.sidebar.button(f"Run {algorithm_name}"):
        mse, r2 = run_algorithm(X, y, algorithm, **kwargs)
        st.write(f"{algorithm_name} Results:")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R² Score: {r2}")

def main():
    st.title("Predictive Algorithms")

    # Load datasets
    cc_data, car_data, student_data = load_data()

    # Dataset selection
    dataset = st.sidebar.selectbox("Select Dataset", ("Select Dataset", "Credit Card", "Sport Car Price", "Student Performance"))

    if dataset == "Credit Card":
        feature = 'Total_Credit_Cards'
        target = 'Avg_Credit_Limit'
        display_results("Credit Card Customer", cc_data, feature, target, "Linear Regression", LinearRegression)
        k = st.sidebar.number_input("Enter k for KNN", min_value=1, value=3)
        display_results("Credit Card Customer", cc_data, feature, target, "KNN", KNeighborsRegressor, n_neighbors=int(k))
        display_results("Credit Card Customer", cc_data, feature, target, "Decision Tree", DecisionTreeRegressor)

    elif dataset == "Sport Car Price":
        feature = 'Horsepower'
        target = 'Price'
        display_results("Sport Car Price", car_data, feature, target, "Linear Regression", LinearRegression)
        k = st.sidebar.number_input("Enter k for KNN", min_value=1, value=3)
        display_results("Sport Car Price", car_data, feature, target, "KNN", KNeighborsRegressor, n_neighbors=int(k))
        display_results("Sport Car Price", car_data, feature, target, "Decision Tree", DecisionTreeRegressor)

    elif dataset == "Student Performance":
        feature = 'studytime'
        target = 'G3'
        display_results("Student Performance", student_data, feature, target, "Linear Regression", LinearRegression)
        k = st.sidebar.number_input("Enter k for KNN", min_value=1, value=3)
        display_results("Student Performance", student_data, feature, target, "KNN", KNeighborsRegressor, n_neighbors=int(k))
        display_results("Student Performance", student_data, feature, target, "Decision Tree", DecisionTreeRegressor)

if __name__ == "__main__":
    main()
