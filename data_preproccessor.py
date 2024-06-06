import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def preprocess_data(data):
    df = pd.DataFrame(data)  # Convert JSON to DataFrame after loop
    df = df[df['city'] == 'Новосибирск']
    df = df.drop(columns=['city', 'id', 'bdAddTime', 'ad_squares', 'anal_type', 'data_range', 'data_date',
                          'data_table'])  # Drop unnecessary columns

    # Convert column values to float32
    df = df.astype('float32')

    # Change the order of columns in the DataFrame df to make the "medial_m" column first
    df = df[
        ['medial_m', 'min_price_m', 'max_price_m', 'average_chronological_m', 'average_arithmetic_m',
         'average_medial_m',
         'upper_medial_m', 'lower_medial_m', 'popular_square', 'min_price_ad', 'max_price_ad',
         'average_chronological_ad',
         'average_arithmetic_ad', 'average_medial_ad', 'medial_ad', 'upper_medial_ad', 'lower_medial_ad', 'price_sum_m',
         'price_sum_ad', 'ad_quantity', 'inPlace']]

    # Extract actual values before normalization
    actual_values = df['medial_m'].values[-30:]

    # Convert to float32
    data = df.values.astype('float32')

    # Split data into X and y
    X = data[:, 1:]  # All features except the first column (target variable)
    y = data[:, 0]  # First column is the target variable

    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create longer time windows
    TIME_STEPS = min(30, len(X))  # Choose the window size

    X_data, y_data = create_dataset(X, y, TIME_STEPS)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X