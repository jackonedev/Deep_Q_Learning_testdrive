from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD


# N_HIDDEN = 128
N_HIDDEN = 512

# Multi Layer Perceptron
def create_model(n_input, n_output, lr, mom, name):

    model = Sequential(name=name)

    # Capa de entrada
    model.add(Dense(units=n_input, activation='relu', input_shape=(n_input,), name='input_layer'))
    model.add(Dropout(0.33))


    # Capa oculta 2
    model.add(Dense(units=N_HIDDEN, activation='relu', name='hidden_layer_2'))
    model.add(Dropout(0.33))

    # Capa de salida
    model.add(Dense(units=n_output, activation='softmax', name='output_layer'))

    # Compilación del modelo
    optimizer = SGD(learning_rate=lr, momentum=mom)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
