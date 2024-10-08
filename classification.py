import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from loguru import logger
from imblearn.over_sampling import SMOTE


def ingest_and_prep_data(bank_dataset: str = 'data/bank.csv') -> tuple[pd.DataFrame,
                                                                            pd.DataFrame,
                                                                            pd.DataFrame,
                                                                            pd.DataFrame]:
    """
    Ingest and prepare the data for the analysis.
    
    """
    df = pd.read_csv(bank_dataset,  delimiter=';',
                     decimal=',')
    print(df.head())
    feature_cols = ['job', 'marital', 'education',
                    'housing', 'loan', 'default', 'day']
    try:
        X = df[feature_cols].copy()
    except Exception as e:
        logger.error(f'Error: {e}')
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    enc = OneHotEncoder(handle_unknown='ignore')
    X_train_enc = enc.fit_transform(X_train)    
    return X_train_enc, X_test, y_train, y_test


def rebalance_classes(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebalance the classes using SMOTE.
    """
    sm = SMOTE()
    X_balanced, y_balanced = sm.fit_resample(X, y)
    return X_balanced, y_balanced


def get_hyperparam_grid() -> dict:
    """
    Get the hyperparameter grid for the model.
    """
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
       
    }
    return random_grid



def get_randomised_rf_cv(random_grid: dict) -> RandomizedSearchCV:
    """
    Get the RandomizedSearchCV object for the model.
    """
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf,
                                param_distributions=random_grid,
                                n_iter=100,
                                cv=3,
                                verbose=2,
                                random_state=42,
                                scoring='f1',
                                n_jobs=-1)
    
    return rf_random


if __name__ == '__main__':
    X_train_enc, X_test, y_train, y_test = ingest_and_prep_data()
    print(X_train_enc.shape, X_test.shape, y_train.shape, y_test.shape)
    X_balanced, y_balanced = rebalance_classes(X_train_enc, y_train)
    rf_random = get_randomised_rf_cv(get_hyperparam_grid())
    rf_random.fit(X_balanced, y_balanced)
    print(rf_random.best_params_)
    print(rf_random.best_score_)
    # y_pred = rf_random.predict(X_test)
    # print(sklearn.metrics.f1_score(y_test, y_pred))
    
    
    
    