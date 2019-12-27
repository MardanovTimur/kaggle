import numpy as np
from keras.engine.sequential import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from pymystem3 import Mystem

from utils import (create_dataset, get_callbacks, preprocess, read_text,
                   save_labels, save_tokenizer)

mystem = Mystem()

FILE_NAME = 'data/1.txt'
FILE_NAME2 = 'data/2.txt'
EMBEDDING_N_DIM = 32


def build_model(vocab_size, window=3):
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_N_DIM, input_length=window))
    model.add(Dropout(0.1))
    model.add(LSTM(256,
                   dropout=0.1,
                   recurrent_dropout=0.1))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', ],
    )
    return model


if __name__ == '__main__':
    text = read_text([FILE_NAME, FILE_NAME2])
    text = preprocess(text, force=True)

    max_words = 100000000
    WINDOW = 4

    tokenizer = Tokenizer(
        num_words=max_words,
        filters='"#$%&()*+-/:;<=>@[\]^_`{|}~'
    )
    tokenizer.fit_on_texts(text)

    X_train = tokenizer.texts_to_sequences(text)
    print('Train shape:', np.array(X_train).shape)
    X_train_time, Y_train_time = create_dataset(np.array(X_train), WINDOW)

    vocab_size = len(tokenizer.word_index) + 1

    y = to_categorical(Y_train_time, num_classes=vocab_size)

    print('Vocabulary size: ', vocab_size)

    save_labels('models/labels.dump', Y_train_time, y)
    save_tokenizer(FILE_NAME)

    model = build_model(vocab_size, WINDOW)

    model.fit(X_train_time, y,
              epochs=50,
              batch_size=16,
              validation_split=0.1,
              verbose=1,
              callbacks=get_callbacks(),
              shuffle=False)
