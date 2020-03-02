
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization, Activation, Dense, Dropout,
    Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
)
from tensorflow.keras.initializers import Constant


def gradient_kernel(shape, dtype=K.floatx()):
    print(shape)
    if shape[:2] != (3, 3):
        raise ValueError('Currently only supports kernels of shape (3, 3)')
    grad_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    grad_x = grad_x / grad_x.std()

    grad_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    grad_y = grad_y / grad_y.std()

    output = np.zeros(shape)
    for i in range(shape[-1]):
        for j in range(shape[-2]):
            if np.random.uniform() < 0.5:
                output[:, :, j, i] = grad_x
            else:
                output[:, :, j, i] = grad_y
            
    return output


def conv2d_block(input_tensor, n_filters, kernel=gradient_kernel, kernel_size=3, batchnorm=True, name='Conv2D'):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel,
               padding='same', name=name+'_1')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel,
               padding='same', name=name+'_2')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def unet_4x(input_img, n_filters=16, dropout=0.5, batchnorm=True, logits=False):
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    if logits:
        outputs = Conv2D(1, (1, 1), activation=None)(c9)
    else:
        outputs = Conv2D(1, (1, 1), activation='sigmoid', bias_initializer=Constant(value=-np.log((1 - 0.01)/0.01)))(c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


def unet_2x(input_img, n_filters=16, dropout=0.5, batchnorm=True, logits=False):
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, name='c1')
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, name='c2')
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c5 = conv2d_block(p2, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, name='c5')

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c5)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='c8')

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='c9')
    
    if logits:
        outputs = Conv2D(1, (1, 1), activation=None)(c9)
    else:
        outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='truncated_normal', bias_initializer=Constant(value=-np.log((1 - 0.01)/0.01)))(c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model
