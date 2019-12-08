from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

# !pip install -q tensorflow-hub
# !pip install -q tensorflow-datasets
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import csv
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import storage

import functools


# print("Version: ", tf.__version__)
# print("Eager mode: ", tf.executing_eagerly())
# print("Hub version: ", hub.__version__)
# print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.

def get_data():
    with open('../get_sentiment_python/data/labeled_lyrics.csv','r') as csvinput:
        reader = pd.read_csv(csvinput)
        return reader


# normalize labels to match that of the expected format: 
# 0 is negative sentiment, 1 is positive sentiment
def normalize_labels(x):
    if x < 0:
        x = 0
    else: 
        x = 1
    return x
        

def split_data(data):
    train_data = data.iloc[:40, :] # training data
    validation_data = data.iloc[40:80, :] # testing data
    test_data = data.iloc[80:, :]

    train_data['sentiment'] = train_data['sentiment'].apply(normalize_labels)
    validation_data['sentiment'] = validation_data['sentiment'].apply(normalize_labels)
    test_data['sentiment'] = test_data['sentiment'].apply(normalize_labels)

    return train_data, validation_data, test_data

def convert_data_to_tf_dataset(data):
    return (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(data['lyrics'].values, tf.string),
                tf.cast(data['sentiment'].values, tf.float64)
            )
        )
    )

def run():
    data = get_data()
    train_data, validation_data, test_data = split_data(data)

    training_dataset = convert_data_to_tf_dataset(train_data)
    validation_dataset = convert_data_to_tf_dataset(validation_data)
    testing_dataset = convert_data_to_tf_dataset(test_data)

    train_examples_batch, train_labels_batch = next(iter(training_dataset.batch(10)))

    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                            dtype=tf.string, trainable=True)
    hub_layer(train_examples_batch[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    history = model.fit(training_dataset.shuffle(40).batch(12),
        epochs=20,
        validation_data=validation_dataset.batch(12),
        verbose=1)

    results = model.evaluate(testing_dataset.batch(12), verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))
    predictions = model.predict(tf.convert_to_tensor(['Something super duper sad and depressing right here! So depressing!']))
    print(predictions)

run()