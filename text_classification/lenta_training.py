#!/usr/bin/env python
# coding: utf-8

import keras
import pandas as pd
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from utils import (clean_text, get_callbacks, get_dataframe, load_dumped,
                   save_dump)

# APP constants
CSV_FILENAME = 'data/lenta-ru-news.csv'
TOKENIZER_DUMP_FILENAME = 'data/tokenizer_lenta.dump'
X_DATA_DUMP_FILENAME = 'data/x_without_vse_padded_indexes.dump'


# Model constants
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 1254
EMBEDDING_DIM = 100


MODEL_KWARGS = {
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
    'metrics': ['accuracy', ],
}


TRAIN_PARAMS = {
    'epochs': 10,
    'batch_size': 1024,
}


def get_padded_texts(cached=True):
    if cached:
        X = load_dumped(X_DATA_DUMP_FILENAME)
    else:
        tokenizer = load_dumped(TOKENIZER_DUMP_FILENAME)
        cleaned_texts = load_dumped('data/x_text_dumped_array.dump')
        X = tokenizer.texts_to_sequences(cleaned_texts)
        X = keras.preprocessing.sequence.pad_sequences(
            X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)
    print('Text to sequences completed')
    return X


def get_model(input_length, output_length, model_kwargs):
    layers = [
        Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=input_length),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(output_length, activation='softmax'),
    ]
    print('input output length: ', input_length, output_length)
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(**model_kwargs)
    return model


def fit(model, X, Y, **kwargs):
    model.fit(X, Y, **kwargs)


def save_dumps(df):
    x = []
    for text in df['text'].values:
        x.append(clean_text(text))
    save_dump(x, 'data/x_without_vse.dump')

    del df
    tokenizer = load_dumped(TOKENIZER_DUMP_FILENAME)

    X = tokenizer.texts_to_sequences(x)
    del x
    X = keras.preprocessing.sequence.pad_sequences(
        X, maxlen=MAX_SEQUENCE_LENGTH)
    save_dump(X, 'data/x_without_vse_padded_indexes.dump')


if __name__ == '__main__':
    df = get_dataframe(CSV_FILENAME)
    #  save_dumps(df)

    Y = pd.get_dummies(df['tags'], ).values
    print('Dummies completed: ', Y.shape)

    tokenizer = load_dumped(TOKENIZER_DUMP_FILENAME)
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    X = get_padded_texts()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1,
                                                        random_state=69)

    model = get_model(X.shape[1], Y.shape[1], model_kwargs=MODEL_KWARGS)

    fit(model, X_train, Y_train,
        epochs=TRAIN_PARAMS['epochs'],
        batch_size=TRAIN_PARAMS['batch_size'],
        callbacks=get_callbacks(),
        validation_split=0.1,
        shuffle=True)
