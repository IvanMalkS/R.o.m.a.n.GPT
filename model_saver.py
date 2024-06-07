import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import clear_session

from model_treiner import create_and_train_model
from prediction_and_visualization import predict_and_calculate_mean_error


def save_latent_states(model, X, output_dir='latent_states'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure X has the correct shape (batch_size, time_steps, num_features)
    if len(X.shape) != 3:
        X = np.expand_dims(X, axis=0)  # Add batch dimension if missing

    latent_states = model.predict(X, batch_size=32)
    np.save(os.path.join(output_dir, 'latent_states.npy'), latent_states)
    print(f"Latent states saved to {os.path.join(output_dir, 'latent_states.npy')}")


def save_best_model(X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X, max_iterations=100):
    # Create directory for saving models, if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    TIME_STEPS = min(30, len(X))  # Choose the window size
    units = 64

    for i in range(max_iterations):
        print(f'Training iteration {i + 1}')

        # Clear the session to avoid clutter from old models and layers
        clear_session()

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

        daily_errors = [(abs(a - p) / a) * 100 for a, p in zip(actual_values, iter_predictions)]
        print(f"Daily percentage errors (Training {i + 1}): {daily_errors}")
        mean_error = np.mean(daily_errors)
        print(f"Mean percentage error (Training {i + 1}): {mean_error}")
        save_latent_states(model, X)

    save_latent_states(model, X)
    print("Final latent states saved.")
