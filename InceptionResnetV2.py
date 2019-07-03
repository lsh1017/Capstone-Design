import numpy as np
from keras.models import Sequential
import keras.layers as layers
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


# """Builds the 8x8 resnet block."""
def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bn=True):
    """
    Function that apply a Conv2D + Batch Norm + Activation -> this is the elementary building block of the model
    """

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)

    # Use of Batch Normalization
    if use_bn:
        x = BatchNormalization(axis=3)(x)

    x = Activation(activation)(x)

    return x


"""The 3 following functions define the 3 recurrent blocks of the Inception_Resnet v2 Network"""

def block35(x, scale, activation='relu'):
    """Builds the 35x35 resnet block."""

    # This block if made of 3 paralell sub-networks

    # 1st sub-network
    branch_0 = conv2d_bn(x, 32, 1)

    # 2nd sub-network
    branch_1 = conv2d_bn(x, 32, 1)
    branch_1 = conv2d_bn(branch_1, 32, 3)

    # 3rd sub-network
    branch_2 = conv2d_bn(x, 32, 1)
    branch_2 = conv2d_bn(branch_2, 48, 3)
    branch_2 = conv2d_bn(branch_2, 64, 3)

    # We concatenate all the outputs of the previous sub-networks
    branches = [branch_0, branch_1, branch_2]
    mixed = Concatenate(axis=3)(branches)

    # We must reshape the residual to the size of the original input
    # in order to add them together
    up = conv2d_bn(mixed, K.int_shape(x)[3], 1, activation=None, use_bn=False)

    # Scaling of the residual activation : avoid the network to 'die'.
    # It helps to stabilize the training according to the original paper.
    # Then we add the scaled residual and block's input.
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale})([x, up])

    x = Activation(activation)(x)

    return x


def block17(x, scale, activation='relu'):
    """Builds the 17x17 resnet block."""

    # This block if made of 3 paralell sub-networks

    # 1st sub-network
    branch_0 = conv2d_bn(x, 192, 1)

    # 2nd sub-network
    branch_1 = conv2d_bn(x, 128, 1)
    branch_1 = conv2d_bn(branch_1, 160, [1, 5]) # to alleviate calculation the 5x5 kernel is done successively
    branch_1 = conv2d_bn(branch_1, 192, [5, 1]) # with a [1, 5] and a [5, 1] kernel

    # We concatenate all the outputs of the previous sub-networks
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3)(branches)

    # We must reshape the residual to the size of the original input
    # in order to add them together
    up = conv2d_bn(mixed, K.int_shape(x)[3], 1, activation=None, use_bn=False)

    # Scaling of the residual activation : avoid the network to 'die'.
    # It helps to stabilize the training according to the original paper.
    # Then we add the scaled residual to the block's input.
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale})([x, up])

    x = Activation(activation)(x)

    return x


def block8(x, scale, activation='relu'):
    """Builds the 8x8 resnet block."""

    # This block if made of 3 paralell sub-networks

    # 1st sub-network
    branch_0 = conv2d_bn(x, 192, 1)

    # 2nd sub-network
    branch_1 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(branch_1, 224, [1, 3]) # to alleviate calculation the 3x3 kernel is done successively
    branch_1 = conv2d_bn(branch_1, 256, [3, 1]) # with a [1, 3] and a [3, 1] kernel

    # We concatenate all the outputs of the previous sub-networks
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3)(branches)

    # We must reshape the residual to the size of the original input
    # in order to add them together
    up = conv2d_bn(mixed, K.int_shape(x)[3], 1, activation=None, use_bn=False)

    # Scaling of the residual activation : avoid the network to 'die'.
    # It helps to stabilize the training according to the original paper.
    # Then we add the scaled residual to the block's input.
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale})([x, up])

    x = Activation(activation)(x)

    return x


def inceptionResNetV2(number_of_class):
    """Builds the 8x8 resnet block."""

    input_tensor = Input(shape=(299, 299, 3), dtype='float32', name='input')

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(input_tensor, 32, 3, strides=2, padding='same')
    x = conv2d_bn(x, 32, 3, padding='same')
    x = conv2d_bn(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='same')
    x = conv2d_bn(x, 192, 3, padding='same')
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3)(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for i in range(1, 11):
        x = block35(x, scale=0.17)


    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=3)(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for i in range(1, 21):
        x = block17(x, scale=0.1)

    # Auxiliary loss to prevent the gradient to vanish given the depth of the network.
    auxiliary_output = AveragePooling2D(3, strides=3, padding='same')(x)
    auxiliary_output = conv2d_bn(auxiliary_output, 128, 1)
    auxiliary_output = conv2d_bn(auxiliary_output, 768, 1)
    auxiliary_output = GlobalAveragePooling2D(name='auxiliary_avg_pool')(auxiliary_output)
    auxiliary_output = Dense(1, activation='sigmoid')(auxiliary_output)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='same')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3)(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = block8(x, scale=0.2)
    x = block8(x, scale=1, activation=None)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.2)(x)

    output_tensor = Dense(number_of_class, activation='softmax')(x)

    # model = Model(img_input, [x, auxiliary_output])
    model = Model(input_tensor, output_tensor)

    return model
