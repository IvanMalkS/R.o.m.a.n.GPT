import time

import model_saver
from data_loder import load_data
from data_preproccessor import preprocess_data

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices:
    #try:
        #for gpu in physical_devices:
            #tf.config.experimental.set_memory_growth(gpu, True)
        #print(f'{len(physical_devices)} GPU(s) available.')
    #except RuntimeError as e:
        #print(e)
#
#else:
    #print('No GPUs found.')

start_time = time.time()

# Load data
data = load_data()

if data:
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X = preprocess_data(data)

    # Define hyperparameters
    time_steps = 30
    units = 64
    batch_size = 32
    max_iterations = 1000

    # Save best model
    model_saver.save_best_model(X_train, X_test, y_train, y_test, scaler_X, scaler_y, actual_values, X, time_steps, units, batch_size, max_iterations)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time:.2f} seconds')
else:
    print("Failed to load data.")
