import math
import optuna
import pandas as pd
from datetime import datetime, timedelta

from warnings import simplefilter

from trade_bot.strategy.lgbm.features_engineering import Features, remove_cols
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score


def train_test_split(
    features_data: pd.DataFrame,
    test_period: timedelta,
    current_time: datetime,
    label_deltatime: timedelta
):
    train_df = features_data[(features_data['time'] < current_time - label_deltatime - test_period)] # & (datetime(2023, 3, 1) <= features_data['time']) & (features_data['time'] <= datetime(2024, 1, 1))]
    test_df = features_data[(features_data['time'] >= current_time - label_deltatime - test_period)] # | ((datetime(2023, 3, 1) <= features_data['time']) & (features_data['time'] <= datetime(2024, 1, 1)))]
    print('Train size: {}, Test size: {}'.format(train_df.shape[0], test_df.shape[0]))
    if test_df.shape[0] == 0:
        print()
    
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df = remove_cols(train_df)
    test_df = remove_cols(test_df)

    return train_df, test_df


def train_model(
    features_data: dict[str, Features],
    test_period: timedelta,
    current_time: datetime,
    label_deltatime: timedelta,
    volume_threshold: float = 5e6
):
    concat_features = [f.features_df for f in features_data.values()]
    concat_features = pd.concat(concat_features)

    concat_features = concat_features[(abs(concat_features['close']/concat_features['open']-1) > 0.04) | (abs(concat_features['high']/concat_features['low']-1) > 0.08) | (abs(concat_features['cur_volume_vs_avg10_diff']) > 1.)]
    concat_features = concat_features[(concat_features['volume']*concat_features['close'] > volume_threshold)].dropna()

    train_df, test_df = train_test_split(concat_features, test_period, current_time, label_deltatime)

    y_train = train_df['label']
    y_test = test_df['label']
    X_train = train_df.drop('label', axis=1)
    X_test = test_df.drop('label', axis=1)
    weight = None
    sample_weight = None
    if 'weight' in X_train.columns:
        weight = X_train['weight']
        sample_weight = X_test['weight']
        X_train = X_train.drop('weight', axis=1)
        X_test = X_test.drop('weight', axis=1)

    def objective(trial):
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)

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
        preds = gbm.predict(X_test)
        auc = roc_auc_score(y_test, preds, sample_weight=sample_weight)
        best_prec = 0
        best_threshold = 0.1
        for t in range(10, 90, 1):
            pred_labels = preds > t/100
            if pred_labels.sum() < 20:
                break
            prec = precision_score(y_test, pred_labels, sample_weight=sample_weight)
            if prec > best_prec:
                best_threshold = t/100
                best_prec = prec
        total_predicted_positive = (preds > best_threshold).sum()
        print('Best threshold:', best_threshold)
        print('Validation set AUC:', auc)
        print('Validation set Precision:', best_prec)
        print('Number of predicted positive labels:', total_predicted_positive)
        if math.isnan(auc):
            auc = 0
        trial.set_user_attr('model', gbm)
        top_threshold = sorted(preds, reverse=True)[10]
        if best_prec <= 0.55 or total_predicted_positive < 10:
            best_threshold = None
        trial.set_user_attr('best_precision', best_prec)
        trial.set_user_attr('best_threshold', best_threshold)
        trial.set_user_attr('top_threshold', top_threshold)
        return (best_prec + auc) / 2

    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_model", value=trial.user_attrs["model"])
            study.set_user_attr(key="best_threshold", value=trial.user_attrs["best_threshold"])
            study.set_user_attr(key="best_precision", value=trial.user_attrs["best_precision"])
            study.set_user_attr(key="top_threshold", value=trial.user_attrs["top_threshold"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, callbacks=[callback])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    best_model = study.user_attrs['best_model']
    best_threshold = study.user_attrs['best_threshold']
    top_threshold = study.user_attrs['top_threshold']
    best_precision = study.user_attrs['best_precision']
    if X_test.shape[0] == 0:
        return None

    y_pred_test = best_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_test)
    print("AUC: ", auc)

    if not (best_precision > 0.55) or not (auc > 0.55):
        best_threshold = None
        
    res = {
        'model': best_model,
        'threshold': best_threshold,
        'top_threshold': top_threshold
    }
    print('Best result:', res)
    return res