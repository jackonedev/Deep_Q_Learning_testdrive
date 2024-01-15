import numpy as np
import random
import os
import pickle
from enum import Enum

from tensorflow import keras
# import cv2
import matplotlib.pyplot as plt

from custom_train_test_split import custom_train_test_split


class DATASETS(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"



DATASET = DATASETS.CIFAR10.value


IMG_SIZE = 100 # Optional
VALIDATION_PROPORTION = 0.2 # default
TESTING_PROPORTION = 0.2 # default
random_state = 2024

DATA_STORAGE = f"datasets/{DATASET}/"


def check_random_image(image_batch, image_resize=None):
    random_index = random.randint(0, len(image_batch))
    random_image = image_batch[random_index]
    # if image_resize:
    #     random_image = cv2.resize(random_image, (image_resize, image_resize))
    print(f"random_image.shape: {random_image.shape}")
    plt.imshow(random_image, cmap='gray')
    plt.show()

    
def main():
    # Load the data
    if DATASET == DATASETS.MNIST.value:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif DATASET == DATASETS.CIFAR10.value:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        raise ValueError("Dataset not implemented")

    X = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    X_aux = X.reshape(X.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    # Creating the sets
    X_train, X_val, X_test, \
        y_train, y_val, y_test = \
            custom_train_test_split(
                X_aux, y,
                val_size=VALIDATION_PROPORTION,
                test_size=TESTING_PROPORTION,
                random_state=random_state,
                verbose=True
    )
            
    X_train = X_train.reshape(X_train.shape[0], *X.shape[1:])
    X_val = X_val.reshape(X_val.shape[0], *X.shape[1:])
    X_test = X_test.reshape(X_test.shape[0], *X.shape[1:])

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
        


if __name__ == "__main__":
    main()
    