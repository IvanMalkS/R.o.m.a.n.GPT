import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = r_num / r_den
    return K.mean(r)

def create_and_train_model(X_train, y_train, X_test, y_test, TIME_STEPS, units, batch_size=32):
    model = Sequential()
    model.add(tf.keras.Input(shape=(TIME_STEPS, X_train.shape[2])))

    # Adding CuDNN optimized Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units, return_sequences=True, kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Adding CuDNN optimized Bidirectional GRU layer
    model.add(Bidirectional(GRU(units, return_sequences=True, kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Adding another CuDNN optimized Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units, kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Adding Dense layers with ReLU activation
    model.add(Dense(100, activation=tf.nn.relu, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(100, activation=tf.nn.relu, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))  # Changing activation function to linear for regression

    # Use mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model.compile(optimizer='adam', loss='mse', metrics=[correlation_coefficient])

    # Custom callback to save the model every 10 epochs
    class CustomModelCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                model.save(f'models/model_epoch_{epoch + 1:02d}.keras')

    custom_checkpoint_callback = CustomModelCheckpoint()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    def scheduler(epoch, lr):
        if epoch < 50:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))

    lr_scheduler = LearningRateScheduler(scheduler)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[custom_checkpoint_callback, early_stopping, checkpoint, lr_scheduler]
    )

    return model, history

# Example usage
# model, history = create_and_train_model(X_train, y_train, X_test, y_test, TIME_STEPS=60, units=50, batch_size=32)
