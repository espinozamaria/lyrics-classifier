#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
from django.db.models import Q
from django.views.generic import TemplateView, ListView
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import HttpResponse

from musixmatch import Musixmatch
musixmatch = Musixmatch(env.musixmatch_token)

from .models import Song

import numpy as np

import tensorflow as tf

# !pip install -q tensorflow-hub
# !pip install -q tensorflow-datasets
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import csv
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import storage

import functools



class HomePageView(APIView):
    def get(self, request, **kwargs):
        something = musixmatch.track_search(q_track=kwargs['song_name'], q_artist=kwargs['artist_name'], page_size=10, page=1, s_track_rating='desc')
        return Response(something)

class SearchResultsView(APIView):
    model = Song

    def get(self, request, **kwargs):

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
        
        def convert_data_to_tf_dataset(data):
            return (
                tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(data['lyrics'].values, tf.string),
                        tf.cast(data['sentiment'].values, tf.float64)
                    )
                )
            )

        def run(lyrics):
            data = get_data()
            data['sentiment'] = data['sentiment'].apply(normalize_labels)

            X_train = data.loc[:49, 'lyrics'].values
            y_train = data.loc[:49, 'sentiment'].values
            X_test = data.loc[49:98, 'lyrics'].values
            y_test = data.loc[49:98, 'sentiment'].values

            print(X_train)

            # Embedding layer
            tokenizer_obj = Tokenizer()
            total_reviews = X_train + X_test

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
            # lyrics = ["happy joy exciting glad happy excited happy"]
            test_samples_tokens = tokenizer_obj.texts_to_sequences(lyrics)
            test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

            # print(model.predict(x=test_samples_tokens_pad))
            prediction = model.predict(x=test_samples_tokens_pad)
            print(prediction)
            return prediction

        print(kwargs['track_id'])
        something = musixmatch.track_lyrics_get(track_id=84622935)
        print(something)
        prediction = run(something)
        return Response(prediction)

    def get_queryset(self):
        query = self.request.GET.get('q')
        print(query)
        somethingElse = self.request.GET.get('post_id', '')

        print(somethingElse)

        something = musixmatch.track_lyrics_get(somethingElse)
        # print('\n \n \n', something['message']['body']['lyrics'])
        print('\n \n \n', something)
        lyrics_deets = something['message']['body']

        return lyrics_deets