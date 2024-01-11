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

from mlp_01_p2 import create_model as cm1
from mlp_02_p2 import create_model as cm2

class SCALING(Enum):
    DEFAULT = "minmax"
    NONE = None
    STANDARD = StandardScaler


MODEL_NAME = "mlp_01_p2"
DATA_STORAGE = "datasets/faults/data/"
SCALER = SCALING.DEFAULT.value
VALIDATION_PROPORTION = 0.2
TESTING_PROPORTION = 0.2
random_state = 2024

EPOCHS = 1e3
BATCH_SIZE = 32

LEARNING_RATE = 0.01
MOMENTUM = 0.9
 
MODEL_BASE_NAME = f"{MODEL_NAME}_{int(EPOCHS)}x{BATCH_SIZE}_{int(time.time())}"
MODELS_DIRNAME = f"models/{MODEL_BASE_NAME}.h5"
HISTORIES_DIRNAME = f"histories/{MODEL_BASE_NAME}.pkl"
LOGS_DIR = f"logs/{MODEL_BASE_NAME}"
    
def main():
    
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
        test_size=TESTING_PROPORTION,
        val_size=VALIDATION_PROPORTION,
        random_state=random_state,
        verbose=False
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
    
    
    # Model creation
    tf.keras.backend.clear_session()
    
    n_input = X_train.shape[1]
    n_output = y_val.shape[1]
    
    learning_rate = LEARNING_RATE
    momentum = MOMENTUM
    
    epochs = int(EPOCHS)
    batch_size = int(BATCH_SIZE)
    
    model = cm2(
        name=MODEL_NAME,
        n_input=n_input,
        n_output=n_output,
        lr=learning_rate,
        mom=momentum
    )
    
    # Defining the callbacks
    tensorboard = TensorBoard(log_dir=LOGS_DIR)
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=1,
        # mode="min",
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=MODELS_DIRNAME,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min"
    )
    
    # Training the model
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, early_stopping, model_checkpoint]
    )
    
    # Saving the history
    os.makedirs(HISTORIES_DIRNAME.split("/")[0], exist_ok=True)
    
    with open(HISTORIES_DIRNAME, "wb") as f:
        pickle.dump(history.history, f)
        

if __name__ == "__main__":
    
    m = multiprocessing.Process(target=main)
    
    #start the program
    m.start()   
    #stop/terminate the program and release the resource
    m.join()    
    
    # End the program
    print("Done!")


# TODO: create a RandomSearch for the hyperparameters
