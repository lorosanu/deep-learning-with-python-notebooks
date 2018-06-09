#!/usr/bin/python3

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Shape of train data: {}".format(train_images.shape))
print("Number of train labels: {}".format(len(train_labels)))
print("Exampes of train labels: {}".format(train_labels))

print("Shape of test data: {}".format(test_images.shape))
print("Number of test labels: {}".format(len(test_labels)))
print("Exampes of test labels: {}".format(test_labels))

# preprocess data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# build the network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# compile the network
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# train the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# evaluate the network
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("Accuracy on test set: {:.2%}".format(test_acc))
