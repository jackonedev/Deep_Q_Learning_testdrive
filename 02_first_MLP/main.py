# imports
import pandas as pd
import multiprocessing
import numpy as np
import random
import time
import os
import pickle
from enum import Enum

# from tqdm import tqdm
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

from custom_train_test_split import custom_train_test_split
from pandapipe.pipelines.estimators import (
    CustomOneHotEncoder,
    CustomScaler,
    CustomImputer
)

from mlp_01_p2 import create_model

class SCALING(Enum):
    DEFAULT = "minmax"
    NONE = None
    STANDARD = StandardScaler


MODEL_NAME = "mlp_01_p2"
DATA_STORAGE = "datasets/faults/data/"
SCALER = SCALING.DEFAULT.value
random_state = 2024
dtime = int(time.time())
MODELS_DIRNAME = f"models/{MODEL_NAME}_{dtime}"
HISTORIES_DIRNAME = f"histories/{MODEL_NAME}_{dtime}"
LOGS_DIR = f"logs/{MODEL_NAME}_{dtime}"

    
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
    validation_size = 0.2
    testing_size = 0.2
    
    X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(
        X, y,
        test_size=testing_size,
        val_size=validation_size,
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
    
    
    # Model creation and training
    tf.keras.backend.clear_session()
    
    n_input = X_train.shape[1]
    n_hidden = 128
    n_output = y_train.shape[1]
    
    learning_rate = 0.01
    momentum = 0.9
    
    epochs = 100
    batch_size = 32
    
    model = create_model(
        name=MODEL_NAME,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
        lr=learning_rate,
        mom=momentum
    )
    
    tensorboard = TensorBoard(log_dir=LOGS_DIR)
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="min",
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=MODELS_DIRNAME + "/model_checkpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min"
    )