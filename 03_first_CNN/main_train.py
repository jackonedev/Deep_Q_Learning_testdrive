# imports
import multiprocessing
from time import time
import os
import pickle
from enum import Enum
from itertools import count
from sklearn.preprocessing import StandardScaler
from pandapipe.pipelines.estimators import CustomScaler

# from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

from main_feed import DATASET, DATA_STORAGE


    

class SCALERS(Enum):
    NONE = None
    DEFAULT = "minmax"
    STANDARD = "standard"



DENSE_LAYERS = [2]
LAYER_SIZE = [64]
CONV_LAYERS = [2]



MODEL_NAME = f"cnn_01_a1_{int(time())}"
SCALER = SCALERS.DEFAULT.value
random_state = 2024
workers = 4
EPOCHS = 30
BATCH_SIZE = 32
# LEARNING_RATE = 0.0045
# MOMENTUM = 0.777

 
def main(verbose=False):
    from cnn_01_a1 import create_models as cm1

    # Loading the data
    with open(DATA_STORAGE + "train_tuple.pkl", "rb") as f:
        X_train, y_train = pickle.load(f)
        
    with open(DATA_STORAGE + "val_tuple.pkl", "rb") as f:
        X_val, y_val = pickle.load(f)
        
    with open(DATA_STORAGE + "test_tuple.pkl", "rb") as f:
        X_test, y_test = pickle.load(f)
        
    
    # Scaling the data    
    if SCALER == SCALERS.NONE.value:
        # In favor of DRY we should have a scaler that does nothing only for defining a "scaler"
        pass
    elif SCALER == "minmax":
        X_train = keras.utils.normalize(X_train)
        X_val = keras.utils.normalize(X_val)
        X_test = keras.utils.normalize(X_test)
    elif SCALER == "stardard":
        scaler = CustomScaler(scaler=StandardScaler)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    if verbose:
        print("Shapes:")
        print(f"X_train.shape: {X_train.shape}")
        print(f"X_val.shape: {X_val.shape}")
        print(f"X_test.shape: {X_test.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"y_val.shape: {y_val.shape}")
        print(f"y_test.shape: {y_test.shape}")
    
    
    
    # # Model creation
    # # tf.keras.backend.clear_session()
    models = cm1(X_train)#TODO: esto es un dict
    
    # # Defining one callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=1,
        mode="min",
        restore_best_weights=True
    )
        
    x = count(1)
    for model_name, model in models.items():
        # UPDATE FILENAMES
        MODEL_BASE_NAME = f"{next(x):02}_{model_name}_{DATASET}"
        
        MODELS_DIRNAME = f"models/{DATASET}/{MODEL_BASE_NAME}.keras"
        HISTORIES_DIRNAME = f"histories/{DATASET}/{MODEL_BASE_NAME}.pkl"
        LOGS_DIR = f"logs/{DATASET}/{MODEL_BASE_NAME}"
    
        # Training the model
        history = model.fit(
            x=X_train,
            y=y_train,
            epochs=int(EPOCHS),
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            workers=workers,
            callbacks=[
                TensorBoard(log_dir=LOGS_DIR),
                early_stopping,
                ModelCheckpoint(
                    filepath=MODELS_DIRNAME,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                    mode="min"
                )
            ]
        )
        
        # Saving the history
        os.makedirs(HISTORIES_DIRNAME.split("/")[0], exist_ok=True)
    
        with open(HISTORIES_DIRNAME, "wb") as f:
            pickle.dump(history.history, f)
            

if __name__ == "__main__":
    
    m = multiprocessing.Process(target=main, args=(True,))
    
    #start the program
    m.start()   
    #stop/terminate the program and release the resource
    m.join()    
    
    # End the program
    print("Done!")


