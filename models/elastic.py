"""Elastic-net logistic regression baseline.

Trains an elastic-net penalised logistic regression model (via scikit-learn)
on the combined ``'All'`` EMR feature set (train + validation), using grid
search with cross-validation to select the best regularisation
hyper-parameters.  The best model is then evaluated on the test split and
results are saved to ``RESULTS_DIR / 'elasticnet' / results.csv``.

This script serves as a strong non-neural baseline and its predictions can
also be used as an input to the late-fusion combination step.

Typical usage::

    python models/elastic.py
"""

import pickle
import os 
import sys 
import pandas as pd
sys.path.append(os.getcwd())

from sklearn                 import metrics
from sklearn.linear_model    import LogisticRegression
from constants.gridsearch    import PARAMETERS
from constants               import *
from sklearn.model_selection import GridSearchCV


def get_data():
    """Load the ``'All'`` EMR features and labels for each split.

    Merges train and validation data into a single development set used for
    grid-search cross-validation.  Test features and their accession numbers
    are kept separate for final evaluation.

    Returns:
        tuple: ``(x_dev, y_dev, x_test, y_test, acc_test)`` where each
        element is a list.  ``x_*`` contain feature vectors, ``y_*`` contain
        binary labels, and ``acc_test`` contains accession number strings.
    """

    # get paths
    all_feature_path = PARSED_EMR_DICT['All']
    all_train_path = all_feature_path / f"All_train.pkl"
    all_val_path = all_feature_path / f"All_val.pkl"
    all_test_path = all_feature_path / f"All_test.pkl"

    # get data
    all_train = pickle.load(open(all_train_path, 'rb'))
    all_val = pickle.load(open(all_val_path, 'rb'))
    all_test = pickle.load(open(all_test_path, 'rb'))

    # get labels
    mapping = pickle.load(open(ACC_2_LABEL,"rb"))

    # parse and populate feature / label list 
    x_dev, y_dev, x_test, y_test, acc_test = [],[],[],[],[] 
    for k,v in all_train.items():
        x_dev.append(v)
        y_dev.append(mapping[k])
    for k,v in all_val.items():
        x_dev.append(v)
        y_dev.append(mapping[k])
    for k,v in all_test.items():
        x_test.append(v)
        y_test.append(mapping[k])
        acc_test.append(k)
    
    return x_dev, y_dev, x_test, y_test, acc_test


def grid_search(x_dev, y_dev):
    """Select the best elastic-net regularisation hyper-parameters.

    Runs :class:`~sklearn.model_selection.GridSearchCV` over the parameter
    grid defined in ``constants.gridsearch.PARAMETERS['elasticnet']`` and
    returns the fitted best estimator.

    Args:
        x_dev (list): Feature vectors for the development set (train + val).
        y_dev (list): Binary labels for the development set.

    Returns:
        sklearn.linear_model.LogisticRegression: Best fitted estimator
        (maximising AUROC).
    """
    clf = LogisticRegression(
        penalty='elasticnet', solver='saga', random_state=0
    )
    param_grid = PARAMETERS['elasticnet']
    gsc = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='roc_auc'
    )
    gsc.fit(x_dev, y_dev)

    return gsc.best_estimator_


def main():
    """Run elastic-net grid search and save test-set predictions."""
    # get features and labels
    x_dev, y_dev, x_test, y_test, acc_test = get_data()

    # predict
    best_clf = grid_search(x_dev, y_dev)
    probs = best_clf.predict_proba(x_test)
    probs = probs[:,1]

    # save results 
    results = {
        PROBS_COL: probs,
        LABELS_COL: y_test,
        ACCESSION_COL: acc_test
    }
    results_df = pd.DataFrame.from_dict(results)
    results_dir = RESULTS_DIR / 'elasticnet' 
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir/ 'results.csv'
    results_df.to_csv(results_path)
    print(f'Results saved at {results_path}')


if __name__ == "__main__":
    main()