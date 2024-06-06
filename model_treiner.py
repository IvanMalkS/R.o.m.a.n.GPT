import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def create_and_train_model(X_train, y_train, X_test, y_test, TIME_STEPS, units, batch_size=32):
    model = Sequential()
    model.add(LSTM(units, input_shape=(TIME_STEPS, X_train.shape[2]), kernel_initializer='orthogonal'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Custom callback to save the model every 10 epochs
    class CustomModelCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                model.save(f'models/model_epoch_{epoch + 1:02d}.keras')

    custom_checkpoint_callback = CustomModelCheckpoint()

    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[custom_checkpoint_callback]
    )

    return model, history
