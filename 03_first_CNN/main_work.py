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


# CIFAR10
# MODEL = "01_cnn_01_a1_1705445437-2-dense-64-layer-2-conv_cifar10.keras"
# MNIST
MODEL = "01_cnn_01_a1_1705524547-1-dense-64-layer-1-conv_mnist.keras"

MODEL_DIR = f"models/{DATASET}"
IMAGE_DIM = pickle.load(open(DATA_STORAGE + "img_dim.pkl", "rb"))

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
}#CIFAR10

CATEGORIES = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9"
}#MNIST


# Cat images
FILES = [
    "datasets/PetImages/Cat/lolarg.jpg",
    "datasets/PetImages/Cat/my_image2.jpg",
    "datasets/PetImages/Cat/beautifull_cat.jpg",
    "datasets/PetImages/Cat/250fc66b.jpg"
] #CIFAR10

# numbers images
FILES = [
    "datasets/Numbers/five.png",
    "datasets/Numbers/three.png",
] #MNIST


def median_binary_array(data):
    central_value = np.median(data)
    bins = [float('-inf'), central_value, float('inf')]
    discretized_array = np.digitize(data, bins) -1

    return discretized_array

def normal_binary_array(data, std_n=1):
    mean_value = np.mean(data)
    std_dev = np.std(data)

    normalized_arr = np.where(np.abs(data - mean_value) < std_dev*std_n, 1, 0)

    return normalized_arr

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
    """
    Preprocesses the images in the given file path.

    Args:
        filepath (str or list): The path(s) to the image file(s).

    Returns:
        numpy.ndarray: The preprocessed image array.
    """
    img_array = [cv2.imread(file) for file in filepath]
    new_array = np.array([cv2.resize(img, (IMAGE_DIM[0], IMAGE_DIM[1])) for img in img_array])
    return new_array

def show_image(filepath, adjust=True):
    if adjust:
        plt.imshow(prepare([filepath])[0])
    else:
        plt.imshow(cv2.imread(filepath))
    plt.show()


if __name__ == "__main__":
    # COMPROBAR SI LAS DIMENSIONES SON CORRECTAS
    if prepare(FILES)[0].shape != IMAGE_DIM:
        try:
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in prepare(FILES)]
            images = [normal_binary_array(img) for img in images]
            images = np.expand_dims(np.squeeze(images), axis=-1)
            if len(images.shape) < 4:
                images = np.expand_dims(images, axis=0)
        except:
            raise ValueError("Image dimention mismatch")
        
    else:
        images = prepare(FILES)
        images = np.squeeze([apply_scaler(image) for image in images])

    model = tf.keras.models.load_model(f"{MODEL_DIR}/{MODEL}")

    prediction = model.predict(images)

    pred_categories = [CATEGORIES[np.argmax(pred)] for pred in prediction]

    plt.imshow(images[1])
    print(pred_categories)

