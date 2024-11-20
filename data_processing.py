import pandas as pd

# Function to load a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to process credit card customer data
def process_credit_card_data(file_path):
    df = load_data(file_path)
    # Example data processing: fill missing values with 0
    df.fillna(0, inplace=True)
    return df

# Function to process student data
def process_student_data(file_path):
    df = load_data(file_path)
    # Handle case sensitivity for column names
    df.columns = df.columns.str.lower()  # Standardize column names to lowercase
    # Example data processing: fill missing 'age' values with the mean age
    if 'Age' in df.columns:  # Ensure 'age' column exists
        df.fillna({'Age': df['Age'].mean()}, inplace=True)
    else:
        raise ValueError("The 'Age' column is missing in the dataset.")
    return df

# Function to process sport car price data
def process_sport_car_data(file_path):
    df = load_data(file_path)
    # Example data processing: drop rows with missing 'Price' values
    df.columns = df.columns.str.lower()  # Standardize column names to lowercase
    if 'Price' in df.columns:  # Ensure 'price' column exists
        df.dropna(subset=['Price'], inplace=True)
    else:
        raise ValueError("The 'Price' column is missing in the dataset.")
    return df
