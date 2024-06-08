import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loder import load_data
from data_preproccessor import preprocess_data
from tensorflow.keras.utils import get_custom_objects


# Define the custom correlation coefficient function
def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.keras.backend.mean(x, axis=0)
    my = tf.keras.backend.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = tf.keras.backend.sum(tf.multiply(xm, ym), axis=0)
    r_den = tf.keras.backend.sqrt(tf.multiply(tf.keras.backend.sum(tf.keras.backend.square(xm), axis=0),
                                              tf.keras.backend.sum(tf.keras.backend.square(ym), axis=0)))
    r = r_num / r_den
    return tf.keras.backend.mean(r)


# Register the custom function
get_custom_objects()['correlation_coefficient'] = correlation_coefficient


def predict_with_saved_model(model_path, data):
    # Load the saved model with custom objects
    model = tf.keras.models.load_model(model_path, custom_objects={'correlation_coefficient': correlation_coefficient})

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X = preprocess_data(data)

    # Prepare data for predicting the next month
    TIME_STEPS = min(30, len(X))

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

    return predictions, actual_values


def predict_next_day(model_path, data):
    # Load the saved model
    model = tf.keras.models.load_model(model_path, custom_objects={'correlation_coefficient': correlation_coefficient})

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X = preprocess_data(data)

    # Time steps (number of previous days used to predict the next day)
    TIME_STEPS = 30

    last_sequence = X[-TIME_STEPS:]  # Use the last TIME_STEPS days to predict the next day
    last_sequence = last_sequence.reshape((1, TIME_STEPS, X.shape[1]))

    # Predict the next day
    next_day_prediction_normalized = model.predict(last_sequence)
    next_day_prediction = scaler_y.inverse_transform(next_day_prediction_normalized)[0, 0]

    return next_day_prediction


def predict_next_week(model_path, data):
    # Load the saved model
    model = tf.keras.models.load_model(model_path, custom_objects={'correlation_coefficient': correlation_coefficient})

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X = preprocess_data(data)

    # Time steps (number of previous days used to predict the next day)
    TIME_STEPS = 30

    last_sequence = X[-TIME_STEPS:]  # Use the last TIME_STEPS days to predict the next day
    last_sequence = last_sequence.reshape((1, TIME_STEPS, X.shape[1]))

    predictions = []

    for _ in range(7):  # Predict for 7 days
        next_day_prediction_normalized = model.predict(last_sequence)
        next_day_prediction = scaler_y.inverse_transform(next_day_prediction_normalized)[0, 0]
        predictions.append(next_day_prediction)

        # Update the sequence for the next prediction
        new_sequence = np.append(last_sequence[:, 1:, :],
                                 np.array(next_day_prediction).reshape(1, 1, 1).repeat(X.shape[1], axis=2), axis=1)
        last_sequence = new_sequence

    return predictions


if __name__ == "__main__":
    data = load_data()

    if data:
        # Use the saved model for predicting new values
        model_path = "best_mean_model_0.517981918528676_20240608_102307.keras"
        predictions, actual_values = predict_with_saved_model(model_path, data)

        # Print the predictions and actual values
        for i, (prediction, actual) in enumerate(zip(predictions, actual_values[:15])):
            error = abs((actual - prediction) / actual) * 100
            print(f"Day {i + 1}: Predicted Value = {prediction:.2f}, Actual Value = {actual:.2f}, Error = {error:.2f}%")

        # Plotting the actual vs predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(actual_values)), actual_values, label='Actual Values')
        plt.plot(range(len(predictions)), predictions, label='Predicted Values (Saved Model)')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Day')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    else:
        print("Failed to load data.")
