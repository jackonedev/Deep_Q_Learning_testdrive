# imports
import pandas as pd
import multiprocessing
import numpy as np
import random
import time
import os
import pickle
from collections import namedtuple
# from tqdm import tqdm
from custom_train_test_split import custom_train_test_split
from sklearn.pipeline import make_pipeline
from pandapipe.pipelines.estimators import (
    CustomOneHotEncoder,
    CustomScaler,
    CustomImputer
)
from sklearn.preprocessing import StandardScaler
from enum import Enum


class SCALING(Enum):
    DEFAULT = "minmax"
    NONE = None
    STANDARD = StandardScaler


MODEL_NAME = "mlp_01_p2"
DATA_STORAGE = "datasets/faults/data/"
SCALER = SCALING.DEFAULT.value
random_state = 2024

    
if __name__ == "__main__":
    
    df = pd.read_csv("datasets/faults.csv")

    # Impute nan with median
    df_aux = CustomImputer().fit_transform(df.loc[:, (df.isna().sum() != 0).values])
    df.loc[:, df_aux.columns] = df_aux

    # One Hot Notation for the target
    one_hot_encoder = CustomOneHotEncoder(columns=["target"])
    one_hot_target = one_hot_encoder.fit_transform(df)

    # Creating population variables
    X = df.drop(columns=["target"])
    y = one_hot_target
    
    # Creating the sets
    X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(
        X, y,
        test_size=0.2,
        val_size=0.2,
        random_state=random_state,
        verbose=True
    )    
    
    # Scaling the data    
    if SCALER == SCALING.DEFAULT.value:
        scaler = CustomScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    elif SCALER == SCALING.STANDARD.value:
        scaler = CustomScaler(scaler=StandardScaler)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    elif SCALER == SCALING.NONE.value:
        pass
    
    
    # Saving the data
    os.makedirs(DATA_STORAGE, exist_ok=True)
    
    data_tuple = (X, y)
    train_tuple = (X_train, y_train)
    val_tuple = (X_val, y_val)
    test_tuple = (X_test, y_test)
    
    with open(DATA_STORAGE + "data_tuple.pkl", "wb") as f:
        pickle.dump(data_tuple, f)
    
    with open(DATA_STORAGE + "train_tuple.pkl", "wb") as f:
        pickle.dump(train_tuple, f)
        
    with open(DATA_STORAGE + "val_tuple.pkl", "wb") as f:
        pickle.dump(val_tuple, f)

    with open(DATA_STORAGE + "test_tuple.pkl", "wb") as f:
        pickle.dump(test_tuple, f)
    
    
    
    
# Environment variables

# Instantiations of classes

# Final variables

# Main loop

# Ensuring initial state

# generating verbose steps

# implementing some actions

# breaking the step loop

# saving the model