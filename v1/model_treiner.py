import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# Define previous_r as a global variable
previous_r = tf.Variable(0.0, dtype=tf.float32)


# Define correlation_coefficient function
def correlation_coefficient(y_true, y_pred):
    global previous_r  # Access the global variable

    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = r_num / r_den

    # Update previous_r
    delta_r = r - previous_r
    K.update(previous_r, r)

    # Check if the change in correlation coefficient is within the specified range
    threshold = 0.005  # 0.5%
    within_range = tf.logical_and(delta_r >= -threshold, delta_r <= threshold)

    # Return 1 if within range, 0 otherwise
    return tf.cast(within_range, dtype=tf.float32)


# Define create_and_train_model function
def create_and_train_model(X_train, y_train, X_test, y_test, TIME_STEPS, units, batch_size=32):
    model = Sequential()
    model.add(tf.keras.Input(shape=(TIME_STEPS, X_train.shape[2])))

    # Adding CuDNN optimized Bidirectional LSTM layer
    model.add(Bidirectional(
        LSTM(units // 2, return_sequences=True, kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Adding another CuDNN optimized Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units // 2, kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Adding Dense layers with LeakyReLU activation
    model.add(Dense(100, kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(100, kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))  # Using linear activation for regression

    # Use mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Adjusted optimizer with a higher initial learning rate and momentum
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='mse', metrics=[correlation_coefficient])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    def scheduler(epoch, lr):
        if epoch < 20:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))

    lr_scheduler = LearningRateScheduler(scheduler)

    history = model.fit(
        X_train, y_train,
        epochs=300,  # Increased the number of epochs
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,  # Set to 1 to monitor training progress
        callbacks=[early_stopping, lr_scheduler]
    )

    return model, history