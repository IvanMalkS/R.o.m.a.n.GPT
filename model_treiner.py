import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, LayerNormalization, Input
from keras.callbacks import EarlyStopping
import os

# Перенаправление стандартного вывода и ошибок в ноль
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def create_and_train_model(X_train, y_train, X_test, y_test, time_steps, units):
    model = Sequential()
    model.add(Input(shape=(time_steps, X_train.shape[2])))
    model.add(Bidirectional(LSTM(units, activation='tanh', return_sequences=True)))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units, activation='tanh', return_sequences=False)))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Создаем объект оптимизатора RMSprop с learning rate 0.001
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test),
                        callbacks=[early_stopping], verbose=0)

    return model, history
