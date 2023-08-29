from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict, cross_val_score
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from mrmr import mrmr_classif

RANDOM_STATE = 42
DEBUG = False
WINDOW_SIZE = 5
NUM_FEATURES = range(5, 12, 2)
DATA_FILES = ('data/data_astro_set1.csv', 'data/data_astro_set2.csv')
CV_SPLIT = 3
OUT_DIR = 'results'
NUM_JOBS=1 if DEBUG else 10

LABEL_COL = 'behavior'
EXPERIMENT_COL = 'astro.ID'
label_maps = [
        {
            'poking': 'synapse',
            'pokcing': 'synapse',
            'pockin': 'synapse',
            'round': 'synapse',
            'scanning': 'kinapse',
            'scan': 'kinapse',
            'dancing': 'kinapse'
            },
        {
            'poking': 'p',
            'pokcing': 'p',
            'pockin': 'p',
            'round': 'p',
            'scanning': 's',
            'scan': 's',
            'dancing': 'd'
            },
        ]


class MrmrSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=1):
        assert(n_features > 0)
        self.n_features = n_features
        self.features = []

    def fit(self, X, y):
        self.features = mrmr_classif(X, y, K=self.n_features, show_progress=False)
        if DEBUG: print(f'MRMR features: {self.features}')
        return self

    def transform(self, X):
        return X[self.features]

class RollingFeatures(BaseEstimator, TransformerMixin):
    COLUMNS = ['Acceleration', 'Acceleration_X', 'Acceleration_Y',
            'Acceleration_Z',
            'Displacement_Delta_Length', 'Displacement_Delta_X',
            'Displacement_Delta_Y', 'Displacement_Delta_Z',
            'Displacement_Length', 'Displacement_X', 'Displacement_Y',
            'Displacement_Z',
            'Distance_To_Nearest_Neighbour',
            'Distance_from_Origin',
            'Position_X', 'Position_Y', 'Position_Z', 'Speed',
            'Time4', 'Time_Since_Track_Start', 'Velocity_Angle_X',
            'Velocity_Angle_Y', 'Velocity_Angle_Z', 'Velocity_X', 'Velocity_Y',
            'Velocity_Z',
            'Average_Distance_To_3_Nearest_Neighbours', 'Average_Distance_To_5_Nearest_Neighbours', 'Average_Distance_To_9_Nearest_Neighbours'
            ]
    KEY_COL = 'key'

    def __init__(self, window:int=3):
        assert int(window) == window, 'Window size must be an integer'
        assert window > 0, 'Window size must be positive'
        assert window%2 == 1, 'Window size must be odd'
        self.window = window

    def extract_statistical_features(self, X: pd.DataFrame):
        #  STAT_COLS = ['Speed', 'Acceleration', 'Displacement_Delta_Length']
        STAT_COLS = self.COLUMNS
        STAT_FUNCS = ['mean', 'var']

        Y = pd.DataFrame(None, index=X.index, columns=[])
        rolling=X.groupby(self.KEY_COL).rolling(window=self.window, center=True)
        for feature in STAT_COLS:
            for func in STAT_FUNCS:
                Y[f'{feature}__{func}'] = getattr(rolling[feature], func)()\
                    .ffill().bfill().reset_index(0, drop=True)
        return Y

    def extract_advanced_features(self, X: pd.DataFrame):
        XYZ_COLS = ['Position_X', 'Position_Y', 'Position_Z']
        SPEED_COL = 'Speed'
        COAST_THRESHOLD = 1/30  # fixed
        FEAT_COLS = ('__displacement', '__track_length', '__coast_coef')

        Y = pd.DataFrame(None, index=X.index, columns=FEAT_COLS)
        win_offset = self.window//2 + 1
        for _, group in X.groupby(self.KEY_COL):
            idx_bfill = []
            for win in group.rolling(window=self.window, center=True):
                win_size = win.shape[0]
                idx_bfill.append(win.index.values[win_size - win_offset])
                if win_size == self.window:
                    norms = np.linalg.norm(win[XYZ_COLS] - win[XYZ_COLS].iloc[0], axis=1)
                    features = (norms[-1],
                                sum(norms),
                                sum(win[SPEED_COL] < COAST_THRESHOLD) / self.window)
                    Y.at[idx_bfill, FEAT_COLS] = features
                    idx_bfill.clear()
        return Y.ffill()

    def fit(self, *_):
        return self

    def transform(self, X):
        if DEBUG: print(f'Window size: {self.window}')
        return pd.concat(axis=1, objs=[
            X[self.COLUMNS],
            self.extract_statistical_features(X),
            self.extract_advanced_features(X)
            ])

    def filter_window_size(self, X):
        X_filtered = X.groupby(self.KEY_COL).filter(lambda x: len(x) >= self.window)
        return X[~X.index.isin(X_filtered.index)]

def dump(pipeline, X, y, file='backup.dat'):
    with open(file, 'wb') as f:
        pickle.dump((pipeline, X, y), f)

def load(file='backup.dat'):
    with open(file, 'rb') as f:
        return pickle.load(f)

def preprocess_data(csv_files, window_size, experiment_column=EXPERIMENT_COL):
    print('Preprocessing data..', flush=True)
    # DropNA is required since some entries have N/A experiment_column
    df = pd.concat([pd.read_csv(csv).dropna() for csv in csv_files], ignore_index=True)
    # drop missing cols
    df.dropna(axis=1, inplace=True)

    rolling = RollingFeatures(window_size)
    # Sequences shorter than the rolling window size will yield NaNs for the added features
    df_dropped = rolling.filter_window_size(df)
    if not df_dropped.empty:
        print(f'!!! Dropped {len(df_dropped)} samples with insufficient sequence size !!!')

    X = rolling.transform(df)
    y = df[LABEL_COL]
    # fix column type mismatch
    exp = df[experiment_column].astype('string') if experiment_column else None
    return X, y, exp

def _score_cv_search(X, y, group, n_splits):
    pip = Pipeline([
        ('selection', MrmrSelector()),
        ('classification', BalancedRandomForestClassifier(random_state=RANDOM_STATE))])
    search = GridSearchCV(estimator=pip,
            param_grid={
                'selection__n_features': NUM_FEATURES,
                },
            cv=GroupKFold(n_splits=n_splits).split(X, y, group), n_jobs=NUM_JOBS)
    search.fit(X, y)
    model = search.best_estimator_
    print(f'*** CV best score: {search.best_score_:.04f}')
    print(f'*** Features:\t{model["selection"].n_features} {model["selection"].features}')

def _score_cv_fixed(X, y, n_features, group, n_splits, labels):
    pip = Pipeline([
        ('selection', MrmrSelector(n_features)),
        ('classification', BalancedRandomForestClassifier(random_state=RANDOM_STATE))])
    cv_split = list(GroupKFold(n_splits=n_splits).split(X, y, group))
    scores = cross_val_score(
                pip, X, y,
                cv=cv_split,
                n_jobs=NUM_JOBS)
    print(f'*** Cross-Validation scores: {scores}, average: {np.mean(scores)}')

def _fit(X, y, n_features):
    pip = Pipeline([
        ('selection', MrmrSelector(n_features)),
        ('classification', BalancedRandomForestClassifier(random_state=RANDOM_STATE))])
    pip.fit(X, y)
    print(f'*** Selected features: {pip["selection"].features}')
    return pip

def score_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    proba_miss = np.max(y_proba[y_pred != y_test], axis=1)
    proba_hit = np.max(y_proba[y_pred == y_test], axis=1)
    score_f1 = f1_score(y_test, y_pred, average='macro')
    score_geom = geometric_mean_score(y_test, y_pred)
    score_acc = model.score(X_test, y_test)
    print(f'*** Test score:\t[F1] {score_f1:.03f} - [Geometric] {score_geom:.03f} - [Accuracy] {score_acc:.03f}')
    return y_pred

def run():
    print(f'* Window size: {WINDOW_SIZE}')

    data_path = Path(f'data/features_w{WINDOW_SIZE}_train.csv')
    if data_path.exists():
        data = pd.read_csv(data_path)
        X_raw, y_raw, exp_raw = data.drop([LABEL_COL, EXPERIMENT_COL], axis=1), data[LABEL_COL], data[EXPERIMENT_COL].astype('string')
    else:
        X_raw, y_raw, exp_raw = preprocess_data(DATA_FILES, WINDOW_SIZE)
        pd.concat(axis=1, objs=[X_raw, y_raw, exp_raw]).to_csv(data_path, index=False)

    for label_map in label_maps[1:]:
        labels = set(label_map.values())
        label_count = len(labels)
        print(f'* Labels: {labels}')

        # TODO fix labeling errors
        y = y_raw.map(label_map).dropna()
        X = X_raw.loc[y.index]
        exp = exp_raw.loc[y.index]

#        print('** Nested CV')
#        cv_test_split = GroupKFold(n_splits=CV_SPLIT[0]).split(X, y, exp)
#        for train_index,test_index in cv_test_split:
#            X_train, y_train, exp_train = X.iloc[train_index], y.iloc[train_index], exp.iloc[train_index]
#            X_test, y_test, group_test = X.iloc[test_index], y.iloc[test_index], exp.iloc[test_index]
#
#            print('* Training set:\n\tClasses:\t{}\n\tExperiments:\t{}'.format(
#                ' - '.join(f'[{label}] {count}' for label,count in y_train.value_counts().items()),
#                exp_train.value_counts().to_list()))
#            print('* Test set:\n\tClasses:\t{}\n\tExperiments:\t{}'.format(
#                ' - '.join(f'[{label}] {count}' for label,count in y_test.value_counts().items()),
#                group_test.value_counts().to_list()))
#    
#            model, _ = _run_inner_cv(X_train, y_train, exp_train, CV_SPLIT[1])
#            score_model(model, X_test, y_test)
#            print()

        for n_features in NUM_FEATURES:
            print(f'** Number of features: {n_features}', flush=True)
            _score_cv_fixed(X, y, n_features, exp, CV_SPLIT, [l.capitalize() for l in labels])
            print(f'*** Fitting on complete dataset..', flush=True)
            model = _fit(X, y, n_features)
            with open(f'{OUT_DIR}/model_w{WINDOW_SIZE}_l{label_count}_f{n_features}.pickle', 'wb') as f:
                pickle.dump(model, f)

        print('-'*50)

if __name__ == '__main__':
    run()
