import os
import numpy as np
import tensorflow as tf
from model_treiner import create_and_train_model
from prediction_and_visualization import predict_and_calculate_mean_error


def save_best_model(X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X, min_error_threshold=0.5, max_iterations=100):
    # Create directory for saving models, if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    TIME_STEPS = min(30, len(X))  # Choose the window size
    units = 128

    for i in range(max_iterations):
        print(f'Training iteration {i + 1}')
        model, _ = create_and_train_model(X_train, y_train, X_test, y_test, TIME_STEPS, units)

        # Reset last_sequence for new predictions
        last_sequence = X[-TIME_STEPS:]
        last_sequence = last_sequence.reshape((1, TIME_STEPS, X.shape[1]))

        iter_predictions = []
        for _ in range(30):  # Predict for the next 30 days
            next_value_normalized = model.predict(last_sequence)
            next_value = scaler_y.inverse_transform(next_value_normalized)[0, 0]
            iter_predictions.append(next_value)

            # Append the predicted value to the sequence and remove the oldest value
            new_sequence = np.append(last_sequence[:, 1:, :],
                                     np.array(next_value).reshape(1, 1, 1).repeat(X.shape[1], axis=2), axis=1)
            last_sequence = new_sequence

        mean_error = np.mean([(abs(a - p) / a) * 100 for a, p in zip(actual_values, iter_predictions)])
        print(f"Mean percentage error (Training {i + 1}): {mean_error:.2f}%")

        # Save the model if it has the least mean error so far or if it meets the threshold
        if os.path.exists('models/best_model_MSK.keras'):
            loaded_best_model = tf.keras.models.load_model('models/best_model_MSK.keras')
            best_mean_error = predict_and_calculate_mean_error(loaded_best_model, X, scaler_X, scaler_y, actual_values)
            if mean_error < best_mean_error or mean_error <= min_error_threshold:
                model_filename = f'models/best_model_MSK_{mean_error:.2f}%.keras'
                model.save(model_filename)
        else:
            if mean_error <= min_error_threshold:
                model_filename = f'models/best_model_MSK_{mean_error:.2f}%.keras'
                model.save(model_filename)

        if mean_error <= min_error_threshold:
            break
