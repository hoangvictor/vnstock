import optuna
import pandas as pd
from datetime import datetime, timedelta

from warnings import simplefilter

from trade_bot.strategy.lgbm.features_engineering import Features
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score


def train_test_split(
    features_data: pd.DataFrame,
    valid_period: timedelta,
    test_period: timedelta,
    current_time: datetime
):
    train_df = features_data[features_data['time'] < current_time - valid_period - test_period]
    valid_df = features_data[(features_data['time'] >= current_time - valid_period - test_period) & (features_data['time'] < current_time - test_period)]
    test_df = features_data[features_data['time'] >= current_time - test_period]

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    if 'next_close' in train_df.columns:
        del train_df['next_close']
        del valid_df['next_close']
        del test_df['next_close']

    rm_cols = ['open', 'high', 'close', 'low', 'volume', 'time', 'label2', 'symbol', 'next_close']
    for c in rm_cols:
        if c in valid_df.columns:
            del test_df[c]
            del train_df[c]
            del valid_df[c]

    return train_df, valid_df, test_df


def train_model(
    features_data: dict[str, Features],
    valid_period: timedelta,
    test_period: timedelta,
    current_time: datetime,
    volume_threshold: float = 5e7
):
    concat_features = [f.features_df for f in features_data.values()]
    concat_features = pd.concat(concat_features)
    concat_features = concat_features[concat_features['volume']*concat_features['close'] > volume_threshold].dropna()

    train_df, valid_df, test_df = train_test_split(concat_features, valid_period, test_period, current_time)

    y_train = train_df['label']
    y_valid = valid_df['label']
    y_test = test_df['label']
    X_train = train_df.drop('label', axis=1)
    X_valid = valid_df.drop('label', axis=1)
    X_test = test_df.drop('label', axis=1)

    def objective(trial):
        dtrain = lgb.Dataset(X_train, label=y_train)

        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
        }
        gbm = lgb.train(param, dtrain)
        preds = gbm.predict(X_valid)
        auc = roc_auc_score(y_valid, preds)
        best_prec = 0
        best_threshold = 0.2
        for t in range(20, 80, 1):
            pred_labels = preds > t/100
            if pred_labels.sum() < 10:
                break
            prec = precision_score(y_valid, pred_labels)
            if prec > best_prec:
                best_threshold = t/100
                best_prec = prec
        print('Best threshold:', best_threshold)
        trial.set_user_attr('model', gbm)
        trial.set_user_attr('best_threshold', best_threshold)
        return (best_prec + auc) / 2

    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_model", value=trial.user_attrs["model"])
            study.set_user_attr(key="best_threshold", value=trial.user_attrs["best_threshold"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1, callbacks=[callback])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    best_model = study.user_attrs['best_model']
    if X_test.shape[0] == 0:
        return None

    y_pred_test = best_model.predict(X_test)

    print("AUC: ", roc_auc_score(y_test, y_pred_test))

    return best_model