# TODO
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras import backend
# from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy


def get_siamese_model(input_shape:tuple):
    """Siamese model architecture.

    This is a simplified version of the original archtecture. No regularization of weights is used.

    Args
    ----
    input_shape: dimension of the input image

    Refs
    ----
        http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    input1 = Input(input_shape, name='input_1')
    input2 = Input(input_shape, name='input_2')

    # Convolutional Neural Network
    convnet = Sequential([
        Conv2D(64, (10,10), activation='relu',
#                input_shape=input_shape,
#                kernel_initializer="random_normal", bias_initializer="zeros",
#                kernel_regularizer=l2(2e-4),
              ),
        MaxPooling2D(),

        Conv2D(128, (7,7), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, (4,4), activation='relu'),
        MaxPooling2D(),

        Conv2D(256, (4,4), activation='relu'),
        Flatten(),

        # most parameters come from the dense layer, originally 4096
        Dense(4096, activation="sigmoid")
    ], name='convnet')

    # Generate the encodings (feature vectors) for the two images
    # Weights are initialized from here
    # This is the "Siamese" part of the architecture!
    encoded1 = convnet(input1)
    encoded2 = convnet(input2)

    # Add a customized layer to compute the absolute difference between the encodings
    l1dist = lambda x:backend.abs(x[0] - x[1])

    # Final output: a dense layer with a sigmoid unit to generate the similarity score
    output = Dense(1, activation='sigmoid', name='output')(
        Lambda(l1dist, name='l1_metric')([encoded1, encoded2])
    )
#     # the following seems to give a meaningful model also:
#     output = Dense(1, activation='sigmoid', name='output')(
#         l1dist([encoded1, encoded2])
#     )

    # Connect the inputs with the outputs and return the model
#     return prediction  # only a tensor but not a model
    return Model(inputs=[input1,input2], outputs=output, name='Siamese_Networks')