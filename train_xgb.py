import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost as xgb


def load_latents_labels(latents_dir, split):
    lat_path = os.path.join(latents_dir, f"latents_{split}sep.npy")
    lab_path = os.path.join(latents_dir, f"labels_{split}sep.npy")
    if not (os.path.exists(lat_path) and os.path.exists(lab_path)):
        return None, None
    X = np.load(lat_path)
    y = np.load(lab_path)
    return X, y


def train_xgboost(latents_dir, model_out, use_val=True, test_on_test=True, random_state=42,
                  use_sample_weights=True, use_smote=False, do_search=False, n_iter_search=30, n_jobs=1):
    X_train, y_train = load_latents_labels(latents_dir, "train")
    if X_train is None:
        raise FileNotFoundError(f"Train latents/labels not found in {latents_dir}")

    X_val, y_val = load_latents_labels(latents_dir, "val")
    X_test, y_test = load_latents_labels(latents_dir, "test")

    # If val missing, create a holdout from train
    if X_val is None and use_val:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # prepare validation features if a val split exists
    X_val_s = None
    if X_val is not None:
        X_val_s = scaler.transform(X_val)
    # Optionally perform SMOTE oversampling on training set
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
        except Exception as e:
            raise RuntimeError("imblearn is required for SMOTE. Install with `pip install imbalanced-learn`") from e
        sm = SMOTE(random_state=random_state)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    # Optionally compute sample weights to counter class imbalance
    sample_weights = None
    if use_sample_weights:
        classes = np.unique(y_train)
        cw = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
        weight_map = {c: w for c, w in zip(classes, cw)}
        sample_weights = np.array([weight_map[c] for c in y_train])

    # If requested, run a randomized hyperparameter search (no early stopping inside CV)
    if do_search:
        param_dist = {
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'n_estimators': [100, 300, 500, 800],
            'max_depth': [3, 4, 6, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0, 5.0],
            'min_child_weight': [1, 3, 5]
        }
        base = xgb.XGBClassifier(use_label_encoder=False, objective='multi:softprob', random_state=random_state)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
        rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter_search, scoring='accuracy', cv=cv,
                                verbose=2, n_jobs=n_jobs, random_state=random_state)
        # pass sample_weight via fit params if computed
        fit_params = {}
        if sample_weights is not None:
            fit_params['sample_weight'] = sample_weights
        rs.fit(X_train_s, y_train, **fit_params)
        print('Best params from search:', rs.best_params_)
        model = rs.best_estimator_
    else:
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            objective='multi:softprob',
            eval_metric="mlogloss",
            n_estimators=200,
            random_state=random_state,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=6,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        eval_set = None
        if use_val and X_val is not None:
            X_val_s = scaler.transform(X_val)
            eval_set = [(X_val_s, y_val)]
            fit_kwargs = {"eval_set": eval_set, "verbose": True}
            if sample_weights is not None:
                fit_kwargs['sample_weight'] = sample_weights
            model.fit(X_train_s, y_train, **fit_kwargs)
        else:
            if sample_weights is not None:
                model.fit(X_train_s, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_s, y_train)

    # Evaluate on validation (if present) and test (if present)
    if use_val and X_val is not None:
        preds_val = model.predict(X_val_s)
        acc_val = accuracy_score(y_val, preds_val)
        print(f"Validation accuracy: {acc_val:.4f}")
        print(classification_report(y_val, preds_val))
        print("Confusion matrix (val):")
        print(confusion_matrix(y_val, preds_val))

    if test_on_test and X_test is not None:
        X_test_s = scaler.transform(X_test)
        preds_test = model.predict(X_test_s)
        acc_test = accuracy_score(y_test, preds_test)
        print(f"Test accuracy: {acc_test:.4f}")
        print(classification_report(y_test, preds_test))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, preds_test))

    # Save model + scaler together
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_out)
    print(f"Saved model+scaler to {model_out}")


if __name__ == "__main__":
    # Simple configuration: edit these variables as needed and run the script.
    latents_dir = "latents"           # directory containing latents_*.npy and labels_*.npy
    out_path = "xgb_model.pkl"       # where to save the trained model + scaler (joblib)
    use_val = True                     # use provided val split if available, otherwise split train
    test_on_test = True                # evaluate on test split if available
    use_sample_weights = True          # compute class-balanced sample weights
    use_smote = True                  # set True to apply SMOTE oversampling (requires imbalanced-learn)
    do_search = True                  # set True to run randomized hyperparameter search (can be slow)
    n_iter_search = 30                 # number of iterations for randomized search
    n_jobs = 1                         # parallel jobs for search (-1 uses all cores)

    print("Running XGBoost training with the following config:")
    print(f"  latents_dir={latents_dir}")
    print(f"  out_path={out_path}")
    print(f"  use_val={use_val}, test_on_test={test_on_test}")
    print(f"  use_sample_weights={use_sample_weights}, use_smote={use_smote}, do_search={do_search}")

    train_xgboost(
        latents_dir,
        out_path,
        use_val=use_val,
        test_on_test=test_on_test,
        random_state=42,
        use_sample_weights=use_sample_weights,
        use_smote=use_smote,
        do_search=do_search,
        n_iter_search=n_iter_search,
        n_jobs=n_jobs,
    )
