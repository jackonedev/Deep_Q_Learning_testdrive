from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD



N_HIDDEN_1 = 128
N_HIDDEN_2 = 512
N_HIDDEN_3 = 128

# MultiLayer Perceptron
def create_model(n_input, n_output, lr, mom, name):

    model = Sequential(name=name)

    # Input Layer
    model.add(Dense(units=n_input,
                    activation='relu',
                    input_shape=(n_input,),
                    name='input_layer'))
    model.add(Dropout(0.33))


    # Hidden Layer 1
    model.add(Dense(units=N_HIDDEN_1,
                    activation='relu',
                    name='hidden_layer_2'))
    model.add(Dropout(0.33))

    # Hidden Layer 2
    model.add(Dense(units=N_HIDDEN_2,
                    activation='relu',
                    name='hidden_layer_3'))
    model.add(Dropout(0.33))
    
    # Hidden Layer 3
    model.add(Dense(units=N_HIDDEN_3,
                    activation='relu',
                    name='hidden_layer_4'))
    model.add(Dropout(0.33))
    
    # Output Layer
    model.add(Dense(units=n_output,
                    activation='softmax',
                    name='output_layer'))

    # Model compilation
    optimizer = SGD(learning_rate=lr, momentum=mom)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
