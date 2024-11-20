#!/usr/bin/env python
# coding: utf-8

import unittest
import pandas as pd
import pytest
import time
from unittest.mock import patch
import matplotlib.pyplot as plt

# Sample function to test
def add(a, b):
    return a + b

# Define the test class
class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)

# Run the test
unittest.main(argv=[''], verbosity=2, exit=False)

# Creating a DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, None, 8]
})

# Fill missing values with a specified value (e.g., 0)
data['B'] = data['B'].fillna(0)

# Check for missing values again
assert data.isnull().sum().sum() == 0, "There are missing values in the dataset!"

# Shape validation
assert data.shape[0] == 4, "Row count mismatch"
assert data.shape[1] == 2, "Column count mismatch"

# Data type validation for column 'A'
assert data['A'].dtype == 'int64', "Column 'A' type mismatch"

# Example model prediction
X_test = [[1, 2], [3, 4]]  # Sample input
model_prediction = [1, 0]  # Example prediction

assert len(model_prediction) == len(X_test), "Prediction length mismatch"

# Model accuracy check
accuracy = 0.85  # Example accuracy
assert accuracy > 0.80, "Model accuracy is below the threshold"

# Mocking external function
def get_external_data():
    return "Real data"

# Correct the patch to refer to the correct module in the same script
@patch('__main__.get_external_data')
def test_mocked_function(mocked_func):
    mocked_func.return_value = "Mocked data"
    result = get_external_data()
    assert result == "Mocked data", "Mocking failed"

# Performance testing (Execution time)
start_time = time.time()

# Simulate a function call
time.sleep(2)  # Simulate delay

elapsed_time = time.time() - start_time
assert elapsed_time < 3, f"Execution time exceeded threshold: {elapsed_time}s"

# Plotting validation
try:
    plt.plot([1, 2, 3, 4])
    plt.show()
    success = True
except Exception as e:
    success = False

assert success, "Plotting failed"
