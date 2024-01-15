from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D
)
def get_params():
    global DENSE_LAYERS, LAYER_SIZE, CONV_LAYERS, MODEL_NAME

    from main_train import DENSE_LAYERS, LAYER_SIZE, CONV_LAYERS
    from main_train import MODEL_NAME



def generate_model_params():
    """
    Generator function that yields combinations of dense layer size, layer size, and convolutional layer size.

    Yields:
        tuple: A tuple containing the dense layer size, layer size, and convolutional layer size.
    """
    for dense_layer in DENSE_LAYERS:
        for layer_size in LAYER_SIZE:
            for conv_layer in CONV_LAYERS:
                yield dense_layer, layer_size, conv_layer

     

def create_models(X):
    
    get_params()
    
    models = {}
    for dense_layer, layer_size, conv_layer in generate_model_params():
        NAME = f"{MODEL_NAME}-{dense_layer}-dense-{layer_size}-layer-{conv_layer}-conv"
        print(NAME)

        model = Sequential()

        model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for _ in range(conv_layer - 1):
            model.add(Conv2D(layer_size, (3, 3)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
    
    
        for _ in range(dense_layer):
            model.add(Dense(layer_size))
            model.add(Activation("relu"))
            
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        
        models[NAME] = model
    
    return models
    