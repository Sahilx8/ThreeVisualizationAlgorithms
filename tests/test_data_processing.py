import unittest
import pandas as pd
from data_processing import process_credit_card_data, process_student_data, process_sport_car_data

class TestDataProcessing(unittest.TestCase):

    def test_credit_card_data_processing(self):
        # Path to the credit card customer dataset
        file_path = '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/Credit Card Customer Data.csv'
        df = process_credit_card_data(file_path)
        # Check if there are any missing values
        self.assertEqual(df.isnull().sum().sum(), 0, "There are missing values in the processed credit card data.")

    def test_student_data_processing(self):
        # Path to the student data dataset
        file_path = '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/student_data.csv'
        
        # Check the columns in the student data before processing
        df = pd.read_csv(file_path)
        print("Student data columns:", df.columns)

        df = process_student_data(file_path)
        # Check if missing 'Age' values were filled
        self.assertEqual(df['Age'].isnull().sum(), 0, "Missing 'Age' values were not filled correctly.")
        print("Columns after processing:", df.columns)
        # Perform assertions
        assert 'Age' in df.columns, "Column 'Age' is missing!"

    def test_sport_car_data_processing(self):
        # Path to the sport car price dataset
        file_path = '/Users/sahil/Downloads/College/Projects/Predictive Project/Datasets/Sport car price.csv'

        # Check the columns in the sport car data before processing
        df = pd.read_csv(file_path)
        print("Sport car data columns:", df.columns)

        df = process_sport_car_data(file_path)
        # Ensure no missing 'Price' values
        self.assertEqual(df['Price'].isnull().sum(), 0, "Missing 'Price' values were not handled correctly.")

if __name__ == "__main__":
    unittest.main()
