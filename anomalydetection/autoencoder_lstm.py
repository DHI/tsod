import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed


def build_model(X_train, dropout_fraction=0.2, size=128):
    timesteps = X_train.shape[1]
    num_features = X_train.shape[2]

    model = Sequential([
        LSTM(size, input_shape=(timesteps, num_features)),
        Dropout(dropout_fraction),
        RepeatVector(timesteps),
        LSTM(size, return_sequences=True),
        Dropout(dropout_fraction),
        TimeDistributed(Dense(num_features))
    ])

    model.compile(loss='mae', optimizer='adam')
    return model


def fit(model, X_train, y_train=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        shuffle=False
    )
    return history


def calculate_loss(X, X_pred):
    """ Caluclate loss used with threshold to detect anomaly. """
    mae_loss = np.mean(np.abs(X_pred - X), axis=1)
    return mae_loss


def detect(X, model, threshold=0.65):
    X_pred = model.predict(X)
    is_anomaly = calculate_loss(X, X_pred) > threshold
    return is_anomaly
