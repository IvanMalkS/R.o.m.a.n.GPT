import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from datetime import datetime

from model_treiner import create_and_train_model


def save_latent_states(model, X, output_dir='latent_states'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure X has the correct shape (batch_size, time_steps, num_features)
    if len(X.shape) != 3:
        X = np.expand_dims(X, axis=0)  # Add batch dimension if missing

    latent_states = model.predict(X, batch_size=32)
    np.save(os.path.join(output_dir, 'latent_states.npy'), latent_states)
    print(f"Latent states saved to {os.path.join(output_dir, 'latent_states.npy')}")

def save_best_model(X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X, time_steps, units, batch_size=32, max_iterations=1000):
    # Create directory for saving models, if it doesn't exist
    if not os.path.exists('models/best_models'):
        os.makedirs('models/best_models')

    best_mean_error = float('inf')  # Initialize with infinity

    for i in range(max_iterations):
        print(f'Training iteration {i + 1}')

        # Clear the session to avoid clutter from old models and layers
        clear_session()

        model, _ = create_and_train_model(X_train, y_train, X_test, y_test, time_steps, units, batch_size)

        # Reset last_sequence for new predictions
        last_sequence = X[-time_steps:]
        last_sequence = last_sequence.reshape((1, time_steps, X.shape[1]))

        iter_predictions = []
        iter_actual_values = []  # Store the actual values for each day
        for _ in range(15):  # Predict for the next 15 days
            next_value_normalized = model.predict(last_sequence)
            next_value = scaler_y.inverse_transform(next_value_normalized)[0, 0]
            iter_predictions.append(next_value)

            # Append the corresponding actual value for the day being predicted
            iter_actual_values.append(actual_values[len(iter_predictions) - 1])

            # Append the predicted value to the sequence and remove the oldest value
            new_sequence = np.append(last_sequence[:, 1:, :],
                                     np.array(next_value).reshape(1, 1, 1).repeat(X.shape[1], axis=2), axis=1)
            last_sequence = new_sequence

        daily_errors = [(abs(a - p) / a) * 100 for a, p in zip(iter_actual_values, iter_predictions)]
        print(f"Daily percentage errors (Training {i + 1}): {daily_errors}")
        mean_error = np.mean(daily_errors)
        print(f"Mean percentage error (Training {i + 1}): {mean_error}")

        if mean_error < best_mean_error:
            best_mean_error = mean_error
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f'best_mean_model_{mean_error}_{timestamp}.keras'
            model_path = os.path.join('models', 'best_models', model_filename)
            try:
                model.save(model_path)
                print(f"Model saved with Mean percentage error: {mean_error}")
            except OSError as e:
                print(f"Failed to save model: {e}")

        if (i + 1) % 10 == 0:  # Save every 10 iterations
            model.save(f'models/best_models/iteration_{i + 1}.keras')
            print(f"Model saved at iteration {i + 1}")

        # Save latent states for the current iteration
        save_latent_states(model, X)

        # Save the actual values for the days being predicted
        np.save(os.path.join('models', 'best_models', f'iter_{i + 1}_actual_values.npy'), np.array(iter_actual_values))

    print("Final latent states saved.")
