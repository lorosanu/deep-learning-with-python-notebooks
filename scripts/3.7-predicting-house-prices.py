#!/usr/bin/python3

import numpy as np
from keras import models, layers
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

print("Shape of training data: {}".format(train_data.shape))
print("Shape of test data: {}".format(test_data.shape))
print("Samples of targets in the training data: {}".format(train_targets[:10]))

# data preprocessing
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(
      64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# k-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100

all_scores = []
for i in range(k):
    print("Processing fold #{}".format(i))

    indices = np.arange(i * num_val_samples, (i + 1) * num_val_samples)

    mask = np.zeros(len(train_data), dtype=bool)
    mask[indices] = True

    val_data = train_data[mask]
    val_targets = train_targets[mask]

    partial_train_data = train_data[~mask]
    partial_train_targets = train_targets[~mask]

    # build, train and evaluate the network
    model = build_model()
    model.fit(
        partial_train_data,
        partial_train_targets,
        epochs=num_epochs,
        batch_size=1,
        verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print("Scores on k-fold cross-validation: {}".format(all_scores))
print("Average score: {}".format(np.mean(all_scores)))
