import sys
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
DATA_PATH=Path('./data/data_test.csv')
FEATURES_PATH=Path(f'./data/features_w{WINDOW_SIZE}_test.csv')

for path in chain(*MODELS.values()):
    if not (MODELS_DIR/path).exists():
        raise RuntimeError(f'Invalid model path: {path}')

for label_map in label_maps:
    labels = set(label_map.values())
    label_count = len(labels)
    print(f'* Labels: {labels}')

    for model_name in MODELS[label_count]:
        path = MODELS_DIR/model_name
        print(f'** Model: {path.stem}', flush=True)
        with open(path, 'rb') as f:
            model: Pipeline = pickle.load(f)
        importance, feats = zip(
            *sorted(zip(model['classification'].feature_importances_,
                        model['selection'].features),
                    reverse=True))
        #print(list(zip(feats, importance)))
        fig, ax=plt.subplots()
        ax.bar(feats, importance)
        ax.set_xticks(feats)
        ax.set_xticklabels(feats, rotation=90)
        plt.show()

    print('-'*50)
