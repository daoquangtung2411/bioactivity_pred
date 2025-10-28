import os
import numpy as np
import pandas as pd
from helper.load_dataset import load_bace_classification
from helper.preprocess import split_train_valid_test
from helper.features import smi_erg
from helper.cal_metrics import classification_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from hyperopt import hp, tpe, fmin, Trials, space_eval
from hyperopt.pyll import scope
from tabulate import tabulate

# Load dataset
bace_class = load_bace_classification()

# Split dataset
train, valid, test = split_train_valid_test(bace_class)
merge = pd.concat((train, valid))

# Generate fingerprint
train_smis = train['SMILES']
valid_smis = valid['SMILES']
test_smis = test['SMILES']
merge_smis = merge['SMILES']
X_train = [smi_erg(smi) for smi in train_smis]
X_valid = [smi_erg(smi) for smi in valid_smis]
X_test = [smi_erg(smi) for smi in test_smis]
X_merge = [smi_erg(smi) for smi in merge_smis]


# Target defined
y_train = train['Class']
y_valid = valid['Class']
y_test = test['Class']
y_merge = merge['Class']

# Hyperparameters tuning with Hyperopt
trials = Trials()

xgb_search_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 300, 5)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
    "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
    "subsample": hp.uniform("subsample", 0.4, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 1.0),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(0.00001), np.log(100)),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(0.001), np.log(1000)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(1.)),
    "booster": hp.choice('booster', ['gbtree', 'gblinear']),
    'tree_method': 'hist',
    'gamma': hp.uniform('gamma', 0., 5.)
}

def xgb_objective(params):
    model = XGBClassifier(
        n_estimators=params['n_estimators'], 
        max_depth=params['max_depth'], 
        min_child_weight=params['min_child_weight'], 
        subsample=params['subsample'], 
        colsample_bytree=params['colsample_bytree'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        learning_rate=params['learning_rate'],
        booster=params['booster'],
        tree_method=params['tree_method'],
        gamma=params['gamma'],
        random_state=42,
        verbosity=0,
        njob=1
    )
    model.fit(X_train, y_train)
    y_valid_hat = model.predict(X_valid)
    f1 = f1_score(y_valid, y_valid_hat)
    return -f1

best_xgb_params = fmin(
    fn=xgb_objective,
    space=xgb_search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)

best_xgb_params = space_eval(xgb_search_space, best_xgb_params)

model = XGBClassifier(**best_xgb_params, probability=True)
model.fit(X_merge, y_merge)
y_train_pred = model.predict(X_merge)
y_train_score = model.predict_proba(X_merge)[:, 1]
y_test_pred = model.predict(X_test)
y_test_score = model.predict_proba(X_test)[:, 1]

# Calculate metrics

train_metrics = classification_metrics(y_merge, y_train_pred, y_train_score)
test_metrics = classification_metrics(y_test, y_test_pred, y_test_score)

# Print

result_header = ['Metrics', 'Train', 'Test']
result_body = [
    ["Accuracy", f'{train_metrics['accuracy']:.4f}', f'{test_metrics['accuracy']:.4f}'],
    ["Recall"],
    ["Overall recall", f'{train_metrics['recall']:.4f}', f'{test_metrics['recall']:.4f}'],
    ["Class 0 recall", f'{train_metrics['0_recall']:.4f}', f'{test_metrics['0_recall']:.4f}'],
    ["Class 1 recall", f'{train_metrics['1_recall']:.4f}', f'{test_metrics['1_recall']:.4f}'],
    ["Precision", '', ''],
    ["Overall precision", f'{train_metrics['precision']:.4f}', f'{test_metrics['precision']:.4f}'],
    ["Class 0 precision", f'{train_metrics['0_precision']:.4f}', f'{test_metrics['0_precision']:.4f}'],
    ["Class 1 precision", f'{train_metrics['1_precision']:.4f}', f'{test_metrics['1_precision']:.4f}'],
    ["AUC-ROC", f'{train_metrics['auc-roc']:.4f}', f'{test_metrics['auc-roc']:.4f}'],
    ["AUC-PRC", f'{train_metrics['auc-prc']:.4f}', f'{test_metrics['auc-prc']:.4f}'],
]

print('XGBoost Classifier results:')
print(f'Best params: {best_xgb_params}')
print(tabulate(result_body, headers=result_header, tablefmt='grid'))

with open('results/bace_class_svm.txt', 'w') as file:
    file.write(f'BACE classfication\n')
    file.write('XGBoost Classifier results:\n')
    file.write(f'Best params: {best_xgb_params}')
    file.write(tabulate(result_body, headers=result_header, tablefmt='grid'))