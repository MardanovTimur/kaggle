import pathlib as pl

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split
)


matplotlib.use('tkagg')


DATA_DIR = pl.Path(__file__).parent / 'data'


def fit():
    df = pd.read_csv(DATA_DIR / 'train.csv')

    X, y = df.loc[:, 'pixel0':].to_numpy(), df['label'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)

    sc_X = StandardScaler()

    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    clf = SVC(gamma=.1, kernel='poly', random_state=0)

    clf.fit(X_train, y_train)

    pickle.dump(clf, open('clf.dump', 'wb'))


def predict():
    df = pd.read_csv(DATA_DIR / 'train.csv')

    X, y = df.loc[:, 'pixel0':].to_numpy(), df['label'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)

    test = pd.read_csv(DATA_DIR / 'test.csv').loc[:, ].to_numpy()

    sc_X = StandardScaler()

    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sc_test = sc_X.transform(test)

    clf = pickle.load(open('clf.dump', 'rb'))

    y_pred = clf.predict(sc_test)

    submission = pd.Series(y_pred, name='Label')
    submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'),
                            submission], axis=1)
    submission.to_csv('final_submission_v1.csv', index=False)


if __name__ == "__main__":
    predict()
