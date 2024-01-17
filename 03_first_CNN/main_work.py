import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from pandapipe.pipelines.estimators import CustomScaler
import pickle
import matplotlib.pyplot as plt


from main_feed import DATASET, DATA_STORAGE
from main_train import SCALER

MODEL = "01_cnn_01_a1_1705445437-2-dense-64-layer-2-conv_cifar10.keras"
CATEGORIES = {
    0: "airplane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "frog",
    6: "dog",
    7: "horse",
    8: "ship/boat",
    9: "truck"
}

CAT_FILES = [
    "datasets/PetImages/Cat/lolarg.jpg",
    "datasets/PetImages/Cat/my_image2.jpg",
    "datasets/PetImages/Cat/beautifull_cat.jpg",
    "datasets/PetImages/Cat/250fc66b.jpg"
]


IMAGE_DIM = pickle.load(open(DATA_STORAGE + "img_dim.pkl", "rb"))


def apply_scaler(images):
    if SCALER == "minmax":
        images = keras.utils.normalize(images)
    elif SCALER == "standard":
        scaler = CustomScaler(scaler=StandardScaler)
        images = scaler.fit_transform(images)
    else:
        raise ValueError("Scaler not implemented")
    return images


def prepare(filepath):
    img_array = [cv2.imread(file) for file in filepath]
    new_array = np.array([cv2.resize(img, (IMAGE_DIM[0], IMAGE_DIM[1])) for img in img_array])
    return new_array

def show_image(filepath, adjust=True):
    if adjust:
        plt.imshow(prepare([filepath])[0])
    else:
        plt.imshow(cv2.imread(filepath))
    plt.show()



model = tf.keras.models.load_model(f"models/{MODEL}")



prediction = model.predict([prepare(CAT_FILES)])

pred_categories = [CATEGORIES[np.argmax(pred)] for pred in prediction]

print(pred_categories)

