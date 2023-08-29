from itertools import chain
from pathlib import Path
import pickle
import pandas as pd

from imblearn.pipeline import Pipeline

from train import LABEL_COL, preprocess_data, WINDOW_SIZE, label_maps, score_model, MrmrSelector, RollingFeatures

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


MODELS_DIR=Path('./results')
MODELS={
    2: [
        'model_w5_l2_f11.pickle',
        ],
    3: [
        'model_w5_l3_f19.pickle',
        ]
    }
#  MODELS={
#      n_labels: [f'model_w5_l{n_labels}_f{n_features}.pickle' for n_features in range(11,26,2)]
#      for n_labels in [2,3]
#      }
#  MODELS={
#      2: ['model_w5_l2_f3.pickle',
#          'model_w5_l2_f5.pickle',
#          'model_w5_l2_f7.pickle',
#          'model_w5_l2_f9.pickle'],
#      3: ['model_w5_l3_f3.pickle',
#          'model_w5_l3_f5.pickle',
#          'model_w5_l3_f7.pickle',
#          'model_w5_l3_f9.pickle'],
#  }
DATA_PATH=Path('./data/data_test.csv')
FEATURES_PATH=Path(f'./data/features_w{WINDOW_SIZE}_test.csv')

for path in chain(*MODELS.values()):
    if not (MODELS_DIR/path).exists():
        raise RuntimeError(f'Invalid model path: {path}')

if FEATURES_PATH.exists():
    data = pd.read_csv(FEATURES_PATH)
    X_raw, y_raw = data.drop([LABEL_COL], axis=1), data[LABEL_COL]
else:
    X_raw, y_raw, _ = preprocess_data([DATA_PATH], WINDOW_SIZE, experiment_column=None)
    pd.concat(axis=1, objs=[X_raw, y_raw]).to_csv(FEATURES_PATH, index=False)

for label_map in label_maps:
    labels = set(label_map.values())
    label_count = len(labels)
    print(f'* Labels: {labels}')

    y = y_raw.map(label_map).dropna()
    X = X_raw.loc[y.index]

    for model_name in MODELS[label_count]:
        path = MODELS_DIR/model_name
        print(f'** Model: {path.stem}', flush=True)
        with open(path, 'rb') as f:
            model: Pipeline = pickle.load(f)
        print(f'** Features : {model["selection"].features}')
        y_pred=score_model(model, X, y)
        cm = confusion_matrix(y, y_pred, normalize='true')
        cm_display = ConfusionMatrixDisplay(cm, display_labels=[l.capitalize() for l in labels]).plot()
        plt.show()

    print('-'*50)
