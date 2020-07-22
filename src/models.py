
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization, Activation, Dense, Dropout,
    Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,
    UpSampling2D
)
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications import VGG16


def gradient_kernel(shape, dtype=K.floatx()):
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
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='conv2d_block1')
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='conv2d_block2')
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c5 = conv2d_block(p2, n_filters=n_filters*16, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='conv2d_block3')

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c5)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='conv2d_block4')

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='conv2d_block5')
    
    if logits:
        outputs = Conv2D(1, (1, 1), activation=None)(c9)
    else:
        outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='truncated_normal', bias_initializer=Constant(value=-np.log((1 - 0.01)/0.01)))(c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


def dilated_2d(input_tensor, n_filters, kernel_size=3, dilation_rate=1, batchnorm=True, name='DilatedConv2D'):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, 
                kernel_initializer='he_normal', padding='same', name=name+'_1')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, 
                kernel_initializer='he_normal', padding='same', name=name+'_2')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def LFE(input_img, n_filters=16, batchnorm=True, logits=False):
    '''
    Based on "Local Feature Extraction", dilated convolutional architecture from
    "Effective Use of Dilated Convolutions for Segmenting Small Object Instances in Remote Sensing Imagery"
    https://arxiv.org/abs/1709.00179
    '''
    # Front-end module
    x = dilated_2d(input_img, n_filters=n_filters, dilation_rate=1, batchnorm=batchnorm, name='fe_1')
    x = dilated_2d(x, n_filters=2*n_filters, dilation_rate=2, batchnorm=batchnorm, name='fe_2')
    x = dilated_2d(x, n_filters=4*n_filters, dilation_rate=3, batchnorm=batchnorm, name='fe_3')
    
    # LFE module
    x = dilated_2d(x, n_filters=4*n_filters, dilation_rate=3, batchnorm=batchnorm, name='lfe_1')
    x = dilated_2d(x, n_filters=2*n_filters, dilation_rate=2, batchnorm=batchnorm, name='lfe_2')
    x = dilated_2d(x, n_filters=n_filters, dilation_rate=1, batchnorm=batchnorm, name='lfe_3')

    # Head module
    x = dilated_2d(x, n_filters=8*n_filters, dilation_rate=1, kernel_size=1, batchnorm=batchnorm, name='head_1')
    if logits:
        outputs = Conv2D(1, (1, 1), activation=None)(x)
    else:
        outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='truncated_normal', bias_initializer=Constant(value=-np.log((1 - 0.01)/0.01)))(x)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


def stacked_multi_scale(input_img, n_filters=16, batchnorm=True, logits=False):
    
    n1 = dilated_2d(input_img, n_filters=n_filters, dilation_rate=1, batchnorm=batchnorm, name='scale_1')
    n2 = dilated_2d(input_img, n_filters=n_filters, dilation_rate=2, batchnorm=batchnorm, name='scale_2')
    n4 = dilated_2d(input_img, n_filters=n_filters, dilation_rate=4, batchnorm=batchnorm, name='scale_4')
    n8 = dilated_2d(input_img, n_filters=n_filters, dilation_rate=8, batchnorm=batchnorm, name='scale_8')

    c = conv2d_block(concatenate([n1, n2, n4, n8]), n_filters=2*n_filters, kernel='he_normal', kernel_size=3, batchnorm=batchnorm, name='conv2d')
    
    if logits:
        outputs = Conv2D(1, (1, 1), activation=None)(c)
    else:
        outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='truncated_normal', bias_initializer=Constant(value=-np.log((1 - 0.01)/0.01)))(c)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
    x = UpSampling2D(size=(factor, factor), interpolation='bilinear')(x)

    return x


def HED(input_img):
    '''
    Implementation of holistically-nested edge detection

    See: https://arxiv.org/abs/1504.06375
    '''
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    b1= side_branch(x, 1) # 256 256 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x) # 128 128 64

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    b2= side_branch(x, 2) # 256 256 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) # 64 64 128

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    b3= side_branch(x, 4) # 256 256 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) # 32 32 256

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x) # 32 32 512
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    b4= side_branch(x, 8) # 256 256 1

    # fuse
    fuse = concatenate([b1, b2, b3, b4], axis=-1)
    fuse = Conv2D(1, (1, 1), 
        padding='same', 
        use_bias=False, 
        activation=None, 
        kernel_initializer=Constant(value=1/5), 
        kernel_regularizer=regularizers.l2(0.0002)
    )(fuse) # 256 256 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(b1)
    o2    = Activation('sigmoid', name='o2')(b2)
    o3    = Activation('sigmoid', name='o3')(b3)
    o4    = Activation('sigmoid', name='o4')(b4)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[input_img], outputs=[o1, o2, o3, o4, ofuse])

    # layers which will have weights set using pretrained VGG16 model
    transfer_layers = [
        'block1_conv1',
        'block1_conv2',
        'block2_conv1',
        'block2_conv2',
        'block3_conv1',
        'block3_conv2',
        'block3_conv3',
        'block4_conv1',
        'block4_conv2',
        'block4_conv3',
    ]
    vgg16 = VGG16()
    for layer_name in transfer_layers:
        weights = vgg16.get_layer(layer_name).get_weights()
        if layer_name == 'block1_conv1':
            # vgg16 is built for 3 channel RGB input images, 
            # so we'll average across the channel axis in the first layer to match our 1 channel input
            weights[0] = weights[0].mean(axis=2, keepdims=True)
        
        model.get_layer(layer_name).set_weights(weights)
        # model.get_layer(layer_name).trainable = False

    return model
