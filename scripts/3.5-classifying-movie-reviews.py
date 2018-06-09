#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, losses, metrics
from keras.datasets import imdb


def decode_text(sample, windex):
    rindex = {value: key for key, value in windex.items()}
    decoded = ' '.join(rindex.get(index - 3, '?') for index in sample)
    return decoded

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
windex = imdb.get_word_index()

print("Maximum word index in training data: {}".format(
  max([max(sequence) for sequence in train_data])))

# data preprocessing
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("Information on the 1st training example")
print("* class label: {}".format(train_labels[0]))
print("* features (word indices): {}".format(train_data[0]))
print("* decoded text: {}".format(decode_text(train_data[0], windex)))
print("* format after vectorization: {}".format(x_train[0]))

# build the network
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# compile the network
model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy])

# train the network
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val))

history_dict = history.history
print("History content: {}".format(history_dict.keys()))

loss = history.history['loss']
acc = history.history['binary_accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_binary_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/3-5_loss.png')

acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/3-5_accuracy.png')

# second attempt
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print("Results after 4 epochs: {}".format(results))

model.predict(x_test)
