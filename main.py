import time

import numpy as np

import model_saver
from data_loder import load_data
from data_preproccessor import preprocess_data
from prediction_and_visualization import predict_and_visualize

start_time = time.time()

# Load data
data = load_data()

if data:
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X = preprocess_data(data)

    # Predict and visualize
    # predict_and_visualize(X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X,  min_error_threshold=6)


    model_saver.save_best_model(X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X,
                                min_error_threshold=1, max_iterations=100000)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Elapsed time: {elapsed_time:.2f} seconds')
else:
    print("Failed to load data.")
