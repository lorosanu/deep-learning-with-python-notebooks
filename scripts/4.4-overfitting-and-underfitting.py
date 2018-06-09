#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, regularizers
from keras.datasets import imdb


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# preprocess data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# build, compile and train a baseline model
original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

original_hist = original_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_test, y_test))

# build, compile and train a smaller model
smaller_model = models.Sequential()
smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(4, activation='relu'))
smaller_model.add(layers.Dense(1, activation='sigmoid'))

smaller_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

smaller_model_hist = smaller_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_test, y_test))

# build, compile and train a bigger model
bigger_model = models.Sequential()
bigger_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
bigger_model.add(layers.Dense(512, activation='relu'))
bigger_model.add(layers.Dense(1, activation='sigmoid'))

bigger_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

bigger_model_hist = bigger_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_test, y_test))

# compare results
epochs = range(1, 21)

original_val_loss = original_hist.history['val_loss']
smaller_val_loss = smaller_model_hist.history['val_loss']
bigger_val_loss = bigger_model_hist.history['val_loss']

original_train_loss = original_hist.history['loss']
bigger_train_loss = bigger_model_hist.history['loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, smaller_val_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.savefig('plots/4-4_original-smaller_val-loss.png')

plt.clf()
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_val_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.savefig('plots/4-4_original-bigger_val-loss.png')

plt.clf()
plt.plot(epochs, original_train_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_train_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()
plt.savefig('plots/4-4_original-bigger_train-loss.png')

# with L2 regularization
l2_model = models.Sequential()
l2_model.add(layers.Dense(
    16, kernel_regularizer=regularizers.l2(0.001),
    activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(
    16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))

l2_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

l2_model_hist = l2_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_test, y_test))

l2_val_loss = l2_model_hist.history['val_loss']

plt.clf()
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_val_loss, 'bo', label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.savefig('plots/4-4_original-regularization_val-loss')

# with dropout
dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))

dpt_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

dpt_model_hist = dpt_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_test, y_test))

dpt_val_loss = dpt_model_hist.history['val_loss']

plt.clf()
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()
plt.savefig('plots/4-4_original-dropout_val-loss')
