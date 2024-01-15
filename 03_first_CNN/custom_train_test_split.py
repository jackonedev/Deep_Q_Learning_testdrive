import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def custom_train_test_split(X, y, val_size=0.2, test_size=0.2, random_state=42, verbose=False):

    skf = KFold(n_splits=2, shuffle=True, random_state=random_state)

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        for train_index, test_index in skf.split(X, y):
            X_1, X_2 = X.iloc[train_index], X.iloc[test_index]
            y_1, y_2 = y.iloc[train_index], y.iloc[test_index]

    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        for train_index, test_index in skf.split(X, y):
            X_1, X_2 = X[train_index], X[test_index]
            y_1, y_2 = y[train_index], y[test_index]
            
    else:
        raise ValueError("X and y must be pandas.DataFrame or numpy.array")
    
    m = X.shape[0]
    training_size = int(m * (1 - test_size) )
    testing_size = m - training_size
    validation_size = int(training_size * val_size)
    training_size -= validation_size
    
    if verbose:
        msg = """
        Sample rows: {} - columns: {}
        Training rows: {}
        Validation rows: {}
        Testing rows: {}
        """.format(
            m, X.shape[1], training_size, validation_size, testing_size
        )
        print(msg)

    X_train_1, X_val, y_train_1, y_val = train_test_split(
        X_1, y_1, test_size=validation_size, random_state=random_state
    )
    X_train_2, X_test, y_train_2, y_test = train_test_split(
        X_2, y_2, test_size=testing_size, random_state=random_state
    )

    X_train = pd.concat([pd.DataFrame(X_train_1), pd.DataFrame(X_train_2)])
    y_train = pd.concat([pd.DataFrame(y_train_1), pd.DataFrame(y_train_2)])

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame)
        assert isinstance(y_val, pd.DataFrame)
        assert isinstance(y_test, pd.DataFrame)
        return X_train, X_val, X_test, y_train, y_val, y_test
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        X_train = X_train.values
        y_train = y_train.values
        return X_train, X_val, X_test, y_train, y_val, y_test


