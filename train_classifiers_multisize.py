#!/usr/bin/env python3
"""
Train multiple sklearn classifiers from latent vectors + label npy files.

Features:
- Loads latents_{split}sep.npy and labels_{split}sep.npy from a latents dir
- Standard scaling, optional SMOTE, optional class/sample weighting
- Optional randomized hyperparameter search per-estimator
- Evaluates on val/test splits and saves each model+scaler as joblib
"""
import os
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

try:
    import imblearn
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

import random

DEFAULT_CLASSIFIERS = [
    'random_forest', 'gradient_boosting', 'hist_gb', 'knn', 'logistic', 'ridge', 'svc'
]


def load_latents_labels(latents_dir, split):
    lat_path = os.path.join(latents_dir, f"latents_{split}.npy")
    lab_path = os.path.join(latents_dir, f"labels_{split}.npy")
    areas_path = os.path.join(latents_dir, f"areas_{split}.npy")
    if not (os.path.exists(lat_path) and os.path.exists(lab_path)):
        return None, None, None
    X = np.load(lat_path)
    y = np.load(lab_path)
    areas = np.load(areas_path)
    print(areas.shape)
    print(areas.mean())
    print(areas.std())
    return X, y, areas


def get_estimators(random_state=42):
    # return mapping name->(estimator, default_param_dist)
    estimators = {}

    

    estimators['hist_gb'] = (
        HistGradientBoostingClassifier(random_state=random_state),
        {
            'max_iter': [100, 300, 500],
            'learning_rate': [0.01, 0.03, 0.1],
            'max_depth': [3, 5, None]
        }
    )

    estimators['knn'] = (
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance']
        }
    )

    estimators['logistic'] = (
        LogisticRegression(random_state=random_state, max_iter=1000),
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    )

    estimators['ridge'] = (
        RidgeClassifier(),
        {
            'alpha': [0.1, 1.0, 10.0]
        }
    )

    estimators['svc'] = (
        SVC(probability=True, random_state=random_state, max_iter=-1),
        {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'class_weight': ['balanced', None]
        }
    )

    return estimators


def train_classifiers(latents_dir, out_dir, classifiers=None, use_val=True, test_on_test=True,
                      random_state=42, use_smote=False, do_grid_search=False, n_jobs=1,
                      n_iter_search=20):

    X_train, y_train, areas_train = load_latents_labels(latents_dir, 'traincubic')
    if X_train is None:
        raise FileNotFoundError(f"Train latents/labels not found in {latents_dir}")

    X_val, y_val, areas_val = load_latents_labels(latents_dir, 'valcubic')
    X_test, y_test, areas_test = load_latents_labels(latents_dir, 'testcubic')

    estimators = get_estimators(random_state=random_state)

    if classifiers is None:
        classifiers = list(estimators.keys())

    os.makedirs(out_dir, exist_ok=True)

    # Define size groups. Assumption: include area==24 in medium and area==48 in medium
    size_groups = {
        'small': lambda a: a <= 24**2,
        'medium': lambda a: (a > 24**2) & (a <= 48**2),
        'large': lambda a: a > 48**2,
    }

    for size_name, mask_fn in size_groups.items():
        print('\n' + '#' * 60)
        print(f"Processing size group: {size_name}")

        # Filter train/val/test by area
        train_idx = np.where(mask_fn(areas_train))[0]
        if train_idx.size == 0:
            print(f"No training samples for size {size_name}, skipping")
            continue
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        X_val_g = None
        y_val_g = None
        if X_val is not None and areas_val is not None:
            val_idx = np.where(mask_fn(areas_val))[0]
            if val_idx.size > 0:
                X_val_g = X_val[val_idx]
                y_val_g = y_val[val_idx]

        X_test_g = None
        y_test_g = None
        if X_test is not None and areas_test is not None:
            test_idx = np.where(mask_fn(areas_test))[0]
            if test_idx.size > 0:
                X_test_g = X_test[test_idx]
                y_test_g = y_test[test_idx]

        # Basic sanity checks
        unique_classes = np.unique(y_tr)
        if unique_classes.size < 2:
            print(f"Not enough classes in training data for size {size_name}, skipping")
            continue

        # Fit scaler on this group's training data
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        X_val_s = None
        if X_val_g is not None:
            X_val_s = scaler.transform(X_val_g)

        # Optional SMOTE
        if use_smote:
            if not IMBLEARN_AVAILABLE:
                raise RuntimeError('imblearn is required for SMOTE. Install with `pip install imbalanced-learn`')
            sm = SMOTE(random_state=random_state)
            X_tr_s, y_tr = sm.fit_resample(X_tr_s, y_tr)

        # For each classifier, train on the group's data and save model with size suffix
        for name in classifiers:
            if name not in estimators:
                warnings.warn(f"Unknown classifier '{name}', skipping")
                continue

            base_est, param_dist = estimators[name]
            print('\n' + '=' * 60)
            print(f"Training classifier: {name} (size={size_name})")

            model = base_est
            # Determine feasible cv folds for stratified splits
            if do_grid_search:
                # compute minimum samples per class to pick cv folds
                try:
                    counts = np.bincount(y_tr.astype(int))
                    min_count = counts[counts > 0].min()
                    n_splits = min(4, int(min_count))
                    if n_splits < 2:
                        print(f"Not enough samples per class for cross-validation in size {size_name}, skipping hyperparameter search")
                        do_search = False
                    else:
                        do_search = True
                except Exception:
                    do_search = True

            else:
                do_search = False

            if do_search:
                from sklearn.model_selection import RandomizedSearchCV
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                rs = RandomizedSearchCV(model, param_distributions=param_dist,
                                        n_iter=n_iter_search, scoring='accuracy', cv=cv,
                                        verbose=1, n_jobs=n_jobs, random_state=random_state)
                rs.fit(X_tr_s, y_tr)
                print('Best params for', name, rs.best_params_)
                model = rs.best_estimator_
            else:
                model.fit(X_tr_s, y_tr)

            # Evaluate
            preds_train = model.predict(X_tr_s)
            acc_train = accuracy_score(y_tr, preds_train)
            print(f"Train accuracy ({name}, {size_name}): {acc_train:.4f}")
            print(classification_report(y_tr, preds_train))
            print('Confusion matrix (train):')
            print(confusion_matrix(y_tr, preds_train))

            if use_val and X_val_s is not None:
                preds_val = model.predict(X_val_s)
                acc_val = accuracy_score(y_val_g, preds_val)
                print(f"Validation accuracy ({name}, {size_name}): {acc_val:.4f}")
                print(classification_report(y_val_g, preds_val))
                print('Confusion matrix (val):')
                print(confusion_matrix(y_val_g, preds_val))

            if test_on_test and X_test_g is not None:
                X_test_s = scaler.transform(X_test_g)
                preds_test = model.predict(X_test_s)
                acc_test = accuracy_score(y_test_g, preds_test)
                print(f"Test accuracy ({name}, {size_name}): {acc_test:.4f}")
                print(classification_report(y_test_g, preds_test))
                print('Confusion matrix (test):')
                print(confusion_matrix(y_test_g, preds_test))

            out_path = os.path.join(out_dir, f"{name}_model_{size_name}.pkl")
            joblib.dump({'model': model, 'scaler': scaler}, out_path)
            print(f"Saved {name} model+scaler to {out_path}")


def main():
    # Simple, hard-coded configuration. Adjust these variables as needed.
    LATENTS_DIR = 'latents'
    OUT_DIR = 'reg_models/classifiers'
    CLASSIFIERS = None  # None = train all
    USE_VAL = True
    TEST_ON_TEST = True
    RANDOM_STATE = 42
    USE_SMOTE = False
    DO_GRID_SEARCH = True
    N_JOBS = 8

    print('Training classifiers with hard-coded config:')
    print({
        'latents_dir': LATENTS_DIR,
        'out_dir': OUT_DIR,
        'classifiers': CLASSIFIERS,
        'use_val': USE_VAL,
        'test_on_test': TEST_ON_TEST,
        'random_state': RANDOM_STATE,
        'use_smote': USE_SMOTE,
    })

    train_classifiers(
        LATENTS_DIR,
        OUT_DIR,
        classifiers=CLASSIFIERS,
        use_val=USE_VAL,
        test_on_test=TEST_ON_TEST,
    random_state=RANDOM_STATE,
    use_smote=USE_SMOTE,
    do_grid_search=DO_GRID_SEARCH,
    n_jobs=N_JOBS,
    )


if __name__ == '__main__':
    main()
