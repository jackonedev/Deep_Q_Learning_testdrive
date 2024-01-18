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

##TEST: ver una imagen como las que entra en el modelo
X_train, y_train = pickle.load(open(DATA_STORAGE + "train_tuple.pkl", "rb"))
X_train = keras.utils.normalize(X_train)

plt.imshow(X_train[0])

##TEST: ver una imagen como las que entra en el modelo


# MODEL = "01_cnn_01_a1_1705445437-2-dense-64-layer-2-conv_cifar10.keras"
MODEL = "01_cnn_01_a1_1705524547-1-dense-64-layer-1-conv_mnist.keras"
MODEL_DIR = f"models/{DATASET}"
# CATEGORIES = {
#     0: "airplane",
#     1: "car",
#     2: "bird",
#     3: "cat",
#     4: "deer",
#     5: "frog",
#     6: "dog",
#     7: "horse",
#     8: "ship/boat",
#     9: "truck"
# }#CIFAR10
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

NUM_FILES = [
    "datasets/Numbers/five.png",
]

# CAT_FILES = [
#     "datasets/PetImages/Cat/lolarg.jpg",
#     "datasets/PetImages/Cat/my_image2.jpg",
#     "datasets/PetImages/Cat/beautifull_cat.jpg",
#     "datasets/PetImages/Cat/250fc66b.jpg"
# ] #CIFAR10


IMAGE_DIM = pickle.load(open(DATA_STORAGE + "img_dim.pkl", "rb"))

def median_binary_array(data):
    central_value = np.median(data)
    bins = [float('-inf'), central_value, float('inf')]
    discretized_array = np.digitize(data, bins) -1

    return discretized_array


def normal_binary_array(data, std_n=1):
    media = np.mean(data)
    std_dv = np.std(data)

    datos_gaussiana = np.random.normal(
        media,
        std_dv*std_n,
        data.shape
    )

    umbral = media
    # categorias = np.array([np.where(datos > umbral, 1, 0) for datos in datos_gaussiana])
    categorias = np.where(datos_gaussiana > umbral, 1, 0)
    
    return categorias

#TODO:
#TODO:
A = np.squeeze(images)

media = np.mean(data)
std_dv = np.std(data)

umbral_1 = media + std_dv
umbral_2 = media - std_dv

# compare_1: Comparar A con umbral_1 -> los 0's == True's -> si es mayor a umbral == True == 0

# compare_2: Comparar A con umbral_2 -> los 0's == Trues' -> si es menor a umbral == True == 0

# en compare_2 -> los 1 van a ser correctos excepto por los ceros de compare_1

# La interseccion entre la mitad superior de compare_2 con la mitad_superior de compare_1


#TODO:
#TODO:



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



model = tf.keras.models.load_model(f"{MODEL_DIR}/{MODEL}")



# COMPROBAR SI LAS DIMENSIONES SON CORRECTAS
if prepare(NUM_FILES)[0].shape != IMAGE_DIM:
    try:
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in prepare(NUM_FILES)]
        images = np.expand_dims(np.squeeze(images), axis=-1)
    except:
        raise ValueError("Image dimention mismatch")
    
else:
    images = prepare(NUM_FILES)
    
images = [apply_scaler(image) for image in images]

# prediction = model.predict([prepare(CAT_FILES)])
prediction = model.predict(images)

pred_categories = [CATEGORIES[np.argmax(pred)] for pred in prediction]

print(pred_categories)

