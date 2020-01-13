import pathlib
import pickle
import re
from datetime import datetime

import pandas as pd
from nltk.corpus import stopwords
from pymystem3 import Mystem

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[0-9a-z#+_]')
STOPWORDS = set(stopwords.words('russian'))

stemmer = Mystem()


def clean_text(text):
    try:
        text = text.lower()
    except Exception:
        # if value is not exists in current row
        text = str(text).lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = " ".join(stemmer.lemmatize(word)[0] for word in text.split()
                    if word not in STOPWORDS)
    return text


def load_dumped(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save_dump(value, filename):
    with open(filename, 'wb') as file:
        return pickle.dump(value, file)


def get_dataframe(filename):
    def year_extraction(row):
        return int(row['date'][0:4])

    df = pd.read_csv(filename)
    df['year'] = df.apply(lambda row: year_extraction(row), axis=1)

    df = df[df['year'] > 1999]
    df = df[df['tags'] != 'Все']
    return df


def get_callbacks(models_path='models'):
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard  # noqa
    MODELS_DIR = pathlib.Path(models_path)
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    model_checkpoint = ModelCheckpoint(
        str(MODELS_DIR /
            'weights-{epoch:02d}-{val_loss:.2f}.hdf5'),
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1
    )
    reduce_lron = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto'
    )
    return [
        tensorboard_callback,
        model_checkpoint,
        reduce_lron,
    ]
