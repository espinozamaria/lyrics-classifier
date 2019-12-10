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
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU
# from tensorflow.keras.layers import Embedding 

import functools

# print("Version: ", tf.__version__)
# print("Eager mode: ", tf.executing_eagerly())
# print("Hub version: ", hub.__version__)
# print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.

def get_data():
    with open('../get_sentiment_python/data/labeled_lyrics.csv','r') as csvinput:
        reader = pd.read_csv(csvinput, encoding='utf-8')
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
    data['sentiment'] = data['sentiment'].apply(normalize_labels)

    train_data = data.iloc[:40, :] # training data
    # validation_data = data.iloc[40:80, :] # testing data
    test_data = data.iloc[40:, :]
    # print(test_data.values)
    

    # train_data['sentiment'] = train_data['sentiment'].apply(normalize_labels)
    # validation_data['sentiment'] = validation_data['sentiment'].apply(normalize_labels)
    # test_data['sentiment'] = test_data['sentiment'].apply(normalize_labels)

    # data['sentiment'] = data['sentiment'].apply(normalize_labels)

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

# def predict(lyrics):
    

# def embed(X_train, X_test):
#     tokenizer_obj = Tokenizer()
#     total_reviews = X_train + X_test
#     tokenizer_obj.fit_on_texts(total_reviews)

#     # pad sequences
#     max_length = max([len(s.split()) for s in total_reviews])

#     # define vocab size
#     vocab_size = len(tokenizer_obj.word_index) + 1

#     X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
#     X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

#     X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
#     X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')


def run():
    data = get_data()
    data['sentiment'] = data['sentiment'].apply(normalize_labels)
    # train_data, validation_data, test_data = split_data(data)

    # train_data = data.iloc[:40, :] # training data
    # test_data = data.iloc[40:, :]
    
    X_train = data.loc[:49, 'lyrics'].values
    y_train = data.loc[:49, 'sentiment'].values
    X_test = data.loc[50:, 'lyrics'].values
    y_test = data.loc[50:, 'sentiment'].values

    print(y_train)

    # print(y_train)

    # Data set format:
    # training_dataset = convert_data_to_tf_dataset(train_data)
    # validation_dataset = convert_data_to_tf_dataset(validation_data)
    # testing_dataset = convert_data_to_tf_dataset(test_data)

    # embedding = embed(training_dataset, testing_dataset)

    # Embedding layer
    tokenizer_obj = Tokenizer()
    total_reviews = X_train + X_test
    # print(total_reviews)
    tokenizer_obj.fit_on_texts(total_reviews)

    # pad sequences
    max_length = max([len(s.split()) for s in total_reviews])

    # define vocab size
    vocab_size = len(tokenizer_obj.word_index) + 1

    X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
    X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')


    EMBEDDING_DIM = 100

    print('Building model...')

    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    print(y_train)
    model.fit(X_train_pad, y_train, batch_size=128, epochs=5, validation_data=(X_test_pad, y_test), verbose=2)

    # Pass in some lyrics: 
    lyrics = ["happy joy exciting glad happy excited happy"]
    test_samples_tokens = tokenizer_obj.texts_to_sequences(lyrics)
    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

    print(model.predict(x=test_samples_tokens_pad))


    # embedding_layer = tf.keras.layers.Embedding(1000, 5)

    # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    # embeddings = embed([training_dataset])
    # https://medium.com/@Currie32/predicting-movie-review-sentiment-with-tensorflow-and-tensorboard-53bf16af0acf

    # print(session.run([embeddings]))
    # #  https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
    # # https://medium.com/tensorflow/building-a-text-classification-model-with-tensorflow-hub-and-estimators-3169e7aa568
    # # https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/
    
    # # Steps: 
    #     # Embed the lyrics
    #     # Use the embedded lyrics as the first layer
    #     # labels are the second layer
    #     # fit the model
    #     # make predictions




    # train_examples_batch, train_labels_batch = next(iter(training_dataset.batch(512)))

    # embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    # hub_layer = hub.KerasLayer(embedding, input_shape=[], 
    #                         dtype=tf.string, trainable=True)
    # hub_layer(train_examples_batch[:3])

    # model = tf.keras.Sequential()
    # model.add(hub_layer)
    # model.add(tf.keras.layers.Dense(16, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # model.summary()

    # model.compile(optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy'])

    # history = model.fit(training_dataset.shuffle(40).batch(512),
    #     epochs=20,
    #     validation_data=validation_dataset.batch(512),
    #     verbose=1)

    # results = model.evaluate(testing_dataset.batch(512), verbose=2)

    # for name, value in zip(model.metrics_names, results):
    #     print("%s: %.3f" % (name, value))
    # predictions = model.predict(tf.convert_to_tensor(['hate super duper sad and depressing right here! So depressing!']))
    # print(predictions)

run()