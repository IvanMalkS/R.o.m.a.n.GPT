import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model_treiner import create_and_train_model


def predict_and_inverse_transform(model, last_sequence, scaler_y):
    next_value_normalized = model.predict(last_sequence)
    next_value = scaler_y.inverse_transform(next_value_normalized)[0, 0]
    return next_value


def predict_and_visualize(X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X,
                          min_error_threshold=1.0):
    # Create directory for saving models, if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Train a single model with 64 LSTM units
    TIME_STEPS = min(30, len(X))  # Choose the window size

    units = 64
    model, history = create_and_train_model(X_train, y_train, X_test, y_test, TIME_STEPS, units)

    # Prepare data for predicting the next month
    TIME_STEPS = min(30, len(X))  # Choose the window size
    last_sequence = X[-TIME_STEPS:]  # Use the last TIME_STEPS elements to predict the next value
    last_sequence = last_sequence.reshape((1, TIME_STEPS, X.shape[1]))

    predictions = []

    for _ in range(15):  # Predict for the next 30 days
        next_value = predict_and_inverse_transform(model, last_sequence, scaler_y)
        predictions.append(next_value)

        # Append the predicted value to the sequence and remove the oldest value
        new_sequence = np.append(last_sequence[:, 1:, :],
                                 np.array(next_value).reshape(1, 1, 1).repeat(X.shape[1], axis=2), axis=1)
        last_sequence = new_sequence

    # Calculate and print percentage errors
    errors = [(abs(a - p) / a) * 100 for a, p in zip(actual_values, predictions)]
    for i, (actual, predicted, error) in enumerate(zip(actual_values, predictions, errors)):
        print(f"Day {i + 1}: Actual Value = {actual:.2f}, Predicted Value = {predicted:.2f}, Error = {error:.2f}%")

    # Plotting the actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(actual_values)), actual_values, label='Actual Values')
    plt.plot(range(len(predictions)), predictions, label='Predicted Values (Training 1)')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    mean_error = np.mean(errors)
    print(f"Mean percentage error (Training 1): {mean_error:.2f}%")

    # Save the model if it has the least mean error so far or if it meets the threshold
    if os.path.exists('models/best_model.keras'):
        loaded_best_model = tf.keras.models.load_model('models/best_model.keras')
        best_mean_error = predict_and_calculate_mean_error(loaded_best_model, X, scaler_X, scaler_y, actual_values)
        if mean_error < best_mean_error or mean_error <= min_error_threshold:
            model_filename = f'models/best_model_{mean_error:.2f}%.keras'
            model.save(model_filename)
    else:
        if mean_error <= min_error_threshold:
            model_filename = f'models/best_model_{mean_error:.2f}%.keras'
            model.save(model_filename)

    # Retrain the model 10 times and plot results
    all_predictions = []

    for i in range(10):
        print(f'Training iteration {i + 1}')
        model, _ = create_and_train_model(X_train, y_train, X_test, y_test, TIME_STEPS, units)

        # Reset last_sequence for new predictions
        last_sequence = X[-TIME_STEPS:]
        last_sequence = last_sequence.reshape((1, TIME_STEPS, X.shape[1]))

        iter_predictions = []
        for _ in range(15):  # Predict for the next 30 days
            next_value = predict_and_inverse_transform(model, last_sequence, scaler_y)
            iter_predictions.append(next_value)

            # Append the predicted value to the sequence and remove the oldest value
            new_sequence = np.append(last_sequence[:, 1:, :],
                                     np.array(next_value).reshape(1, 1, 1).repeat(X.shape[1], axis=2), axis=1)
            last_sequence = new_sequence

        all_predictions.append(iter_predictions)

        mean_error = np.mean([(abs(a - p) / a) * 100 for a, p in zip(actual_values, iter_predictions)])
        print(f"Mean percentage error (Training {i + 2}): {mean_error:.2f}%")

        # Save the model if it has the least mean error so far or if it meets the threshold
        if os.path.exists('models/best_model.keras'):
            loaded_best_model = tf.keras.models.load_model('models/best_model.keras')
            best_mean_error = predict_and_calculate_mean_error(loaded_best_model, X, scaler_X, scaler_y, actual_values)
            if mean_error < best_mean_error or mean_error <= min_error_threshold:
                model_filename = f'models/best_model_{mean_error:.2f}%.keras'
                model.save(model_filename)
        else:
            if mean_error <= min_error_threshold:
                model_filename = f'models/best_model_{mean_error:.2f}%.keras'
                model.save(model_filename)


def predict_and_calculate_mean_error(model, X, scaler_X, scaler_y, actual_values):
    TIME_STEPS = min(30, len(X))  # Choose the window size
    last_sequence = X[-TIME_STEPS:]  # Use the last TIME_STEPS elements to predict the next value
    last_sequence = last_sequence.reshape((1, TIME_STEPS, X.shape[1]))

    predictions = []

    for _ in range(15):  # Predict for the next 15 days
        next_value_normalized = model.predict(last_sequence)
        next_value = scaler_y.inverse_transform(next_value_normalized)[0, 0]
        predictions.append(next_value)

        # Append the predicted value to the sequence and remove the oldest value
        new_sequence = np.append(last_sequence[:, 1:, :],
                                 np.array(next_value).reshape(1, 1, 1).repeat(X.shape[1], axis=2), axis=1)
        last_sequence = new_sequence

    # Calculate and return the mean error
    mean_error = np.mean([(abs(a - p) / a) * 100 for a, p in zip(actual_values, predictions)])
    return mean_error