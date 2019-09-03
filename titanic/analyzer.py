import logging
import pathlib as pl
import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.stochastic_gradient import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

logger = logging.getLogger(__file__)


DATA_PATH = pl.Path(__file__).parent / 'data'


def scale(df, column):
    scaller = MinMaxScaler()
    data = df[column]
    values = scaller.fit_transform(data.to_numpy().reshape(1, -1))
    df[column] = values.reshape(-1)


def standart_scale(df, column):
    imp = SimpleImputer()
    transformed = imp.fit_transform(df[[column]])
    scaller = StandardScaler()
    values = scaller.fit_transform(transformed)
    df[column] = values.reshape(-1)


def pop(columns, *dfs):
    for df in dfs:
        for column in columns:
            df.pop(column)


def ticket_app(ticket):
    tickets = ticket.split(' ')
    for tick in tickets:
        if tick.isnumeric():
            return tick
    return 0


if __name__ == '__main__':
    test_df = pd.read_csv(DATA_PATH / 'test.csv')
    train_df = pd.read_csv(DATA_PATH / 'train.csv')

    y_train = train_df.pop('Survived')

    train_df['Title'] = train_df['Name'].apply(lambda X: re.search(r'[A-Z]{1}[a-z]+\.', X).group(0))
    test_df['Title'] = test_df['Name'].apply(lambda X: re.search(r'[A-Z]{1}[a-z]+\.', X).group(0))

    def classify_title(dataframe_in):
        dataframe_in.loc[:, ['Title']] = dataframe_in['Title'].apply(
            lambda X: X if X in ['Mr.', 'Miss.', 'Mrs.', 'Master.'] else 'Rare')
        return dataframe_in

    def get_cabin_letter(dataframe_in):
        dataframe_in['Cabin'] = dataframe_in['Cabin'].apply(
            lambda X: re.search(
                '[A-Za-z]{1}',
                X).group(0).upper() if isinstance(
                X,
                str) else '?')
        return dataframe_in

    def fill_age_from_masters(df, strategy_in='median'):
        is_master = (df['Title'] == 'Master.')
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy_in)
        df.loc[is_master, 'Age'] = imp.fit_transform(df.loc[is_master][['Age']])
        return df

    def fill_age_from_non_masters(df, strategy_in='median'):
        is_not_master = (df['Title'] != 'Master.')
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy_in)
        df.loc[is_not_master, 'Age'] = imp.fit_transform(df.loc[is_not_master][['Age']])
        return df

    train_df = fill_age_from_masters(train_df)
    train_df.isnull().sum()

    test_df = fill_age_from_masters(test_df)
    test_df.isnull().sum()

    train_df = fill_age_from_non_masters(train_df)
    train_df.isnull().sum()

    test_df = fill_age_from_non_masters(test_df)
    test_df.isnull().sum()

    #  train_df = classify_title(train_df)
    #  test_df = classify_title(test_df)

    train_df = get_cabin_letter(train_df)
    test_df = get_cabin_letter(test_df)

    pop(('Name', 'Ticket', 'Cabin'), test_df, train_df)

    #  train_df['Ticket'] = train_df.apply(lambda x: ticket_app(x['Ticket']), axis=1)
    #  test_df['Ticket'] = test_df.apply(lambda x: ticket_app(x['Ticket']), axis=1)

    print(train_df)



    # Preprocess title
    title_classes = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Rare']
    title_class = dict([(title, val) for title, val in zip(
        title_classes, np.linspace(0, 1, len(title_classes)))])

    train_df['Title'] = train_df['Title'].apply(title_class.get)
    test_df['Title'] = test_df['Title'].apply(title_class.get)

    # Preprocess male-female features
    male_female = {'male': 0, 'female': 1}
    train_df['Sex'] = train_df['Sex'].apply(male_female.get)
    test_df['Sex'] = test_df['Sex'].apply(male_female.get)

    embarked = {'S': .0, 'C': .5, 'Q': 1.0}
    train_df['Embarked'] = train_df['Embarked'].apply(embarked.get)
    test_df['Embarked'] = test_df['Embarked'].apply(embarked.get)

    scalable_rows = (
        'Pclass',
    )
    for row in scalable_rows:
        scale(train_df, row)
        scale(test_df, row)

    scalable_rows = [
        'Fare',
        'Age',
    ]
    for row in scalable_rows:
        standart_scale(train_df, row)
        standart_scale(test_df, row)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    pca = PCA()

    X_train = pca.fit_transform(train_df.loc[:, 'Pclass':].to_numpy())
    X_test = pca.transform(test_df.loc[:, 'Pclass':].to_numpy())

    #  X_test = pca.fit_transform(X_test)

    passenger_id = train_df['PassengerId'].to_numpy(), \
        train_df.loc[:, 'Pclass':].to_numpy()

    passenger_id_test = test_df['PassengerId'].to_numpy()

    parameters_grid = {
        'loss': ['hinge', 'log', 'squared_hinge', 'squared_loss', ],
        'penalty': ['l2', 'l1'],
        'alpha': np.linspace(.0001, .01, num=20),
    }
    hyper_xgboost = {
        'eta': [0, 0.00001],  # 0 = OPT
        'gamma': [0.001, 0.05, 0.1],  # 0.05 =OPT
        'max_depth': [3, 4, 5],  # 4 = OPT
        'probability': [True],
        'random_state': [42]
    }
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.1, random_state=0)

    classifier = xgb.XGBClassifier()

    grid_cv = GridSearchCV(classifier, hyper_xgboost, scoring='accuracy', cv=cv)
    grid_cv.fit(X_train, y_train.to_numpy())
    print(grid_cv.best_score_)
    print(grid_cv.best_params_)

    #  classifier = SGDClassifier(n_jobs=4, **grid_cv.best_params_)
    classifier = xgb.XGBClassifier(**grid_cv.best_params_)
    classifier.fit(X_train, y_train)

    y_test = classifier.predict(X_test)

    submission = pd.Series(y_test, name='Survived')
    submission = pd.concat([pd.Series(passenger_id_test, name='PassengerId'),
                            submission], axis=1)
    submission.to_csv(DATA_PATH / 'final_submission_v4.csv', index=False)

    print(y_test)
