import numpy as np 
import keras
from keras.layers import Conv2D,MaxPooling2D,Dropout,LeakyReLU,concatenate,ZeroPadding2D,BatchNormalization,Conv2DTranspose
from keras.layers import Add
from keras.models import Model
from keras.layers import Input, Dense
import os 
import cv2 
from keras.applications.densenet import DenseNet121
from matplotlib.pyplot import imshow,title ,show
from keras.applications import Xception
from keras import backend as K
from keras.losses import binary_crossentropy as binary_crossentropy
from keras.callbacks import ModelCheckpoint,Callback
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid


from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, regularizers
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import BatchNormalization as bn
from tensorflow.keras.layers import (Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, Dropout, GlobalAveragePooling2D,
                                     GlobalMaxPool2D, GlobalMaxPooling2D,
                                     Input, Lambda, LeakyReLU, MaxPooling2D,
                                     Permute, RepeatVector, Reshape,
                                     UpSampling2D, ZeroPadding2D, add,
                                     concatenate, multiply)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


def UppXceptionWithAttention(input_shape=(None, None, 3), dropout_rate=0.5):
    """
    U-Net with Xception encoder, with CBAM (convolutional block attention module) blocks.

    # TODO: fix functionality to choose between CBAM and SE (squeeze and excitation)


    :param input_shape: shape of the input image
    :type input_shape: tuple
    :param dropout_rate: dropout probability
    :type dropout_rate: float
    :return: TF model 
    """

    def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        #x = attach_attention_module(x,attention_module='cbam_block')
        if activation == True:
            x = LeakyReLU(alpha=0.1)(x)
        return x

    def residual_block(blockInput, num_filters=16):
        x = LeakyReLU(alpha=0.1)(blockInput)
        x = BatchNormalization()(x)
        x = attach_attention_module(x, attention_module='cbam_block')
        blockInput = BatchNormalization()(blockInput)
        x = convolution_block(x, num_filters, (3, 3))
        x = convolution_block(x, num_filters, (3, 3), activation=False)
        x = Add()([x, blockInput])
        #x = attach_attention_module(x,attention_module='cbam_block')
        return x

    def attach_attention_module(net, attention_module):
        if attention_module == 'se_block':  # SE_block
            net = se_block(net)
        elif attention_module == 'cbam_block':  # CBAM_block
            net = cbam_block(net)
        else:
            raise Exception(
                "'{}' is not supported attention module!".format(attention_module))

        return net

    def se_block(input_feature, ratio=8):
        """Contains the implementation of Squeeze-and-Excitation(SE) block.
        As described in https://arxiv.org/abs/1709.01507.
        """

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]

        se_feature = GlobalAveragePooling2D()(input_feature)
        se_feature = Reshape((1, 1, channel))(se_feature)
        assert se_feature.shape[1:] == (1, 1, channel)
        se_feature = Dense(channel // ratio,
                           activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')(se_feature)
        assert se_feature.shape[1:] == (1, 1, channel//ratio)
        se_feature = Dense(channel,
                           activation='sigmoid',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')(se_feature)
        assert se_feature.shape[1:] == (1, 1, channel)
        if K.image_data_format() == 'channels_first':
            se_feature = Permute((3, 1, 2))(se_feature)

        se_feature = multiply([input_feature, se_feature])
        return se_feature

    def cbam_block(cbam_feature, ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """

        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature)
        return cbam_feature

    def channel_attention(input_feature, ratio=8):

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]

        shared_layer_one = Dense(channel//ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def spatial_attention(input_feature):
        kernel_size = 7

        if K.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = Permute((2, 3, 1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(
            x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(
            x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        assert cbam_feature.shape[-1] == 1

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    backbone = Xception(input_shape=input_shape,
                        weights='imagenet', include_top=False)
    input = backbone.input
    start_neurons = 8

    # Encoder
    conv0_0 = backbone.layers[0].output # Input layer
    conv0_0 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same")(conv0_0)
    conv0_0 = residual_block(conv0_0, start_neurons * 2)
    conv0_0 = residual_block(conv0_0, start_neurons * 2)
    conv0_0 = LeakyReLU(alpha=0.1)(conv0_0)

    conv1_0 = backbone.layers[11].output
    conv1_0 = ZeroPadding2D(((3, 0), (3, 0)))(conv1_0)

    conv2_0 = backbone.layers[21].output
    conv2_0 = ZeroPadding2D(((1, 0), (1, 0)))(conv2_0)

    conv3_0 = backbone.layers[31].output

    conv4_0 = backbone.layers[121].output
    conv4_0 = LeakyReLU(alpha=0.1)(conv4_0)

    conv5_0 = Conv2D(start_neurons * 64, (1, 1), activation=None, padding="same")(MaxPooling2D()(conv4_0))
    conv5_0 = residual_block(conv5_0, start_neurons * 64)
    conv5_0 = residual_block(conv5_0, start_neurons * 64)
    conv5_0 = LeakyReLU(alpha=0.1)(conv5_0)

    # Decoder
    conv0_1 = Conv2DTranspose(start_neurons*2, (2,2), strides=(2, 2), padding="same")(conv1_0)
    conv0_1 = concatenate([conv0_1, conv0_0])
    conv0_1 = Conv2D(start_neurons*2, (1,1), activation=None, padding="same")(conv0_1)
    conv0_1 = residual_block(conv0_1, start_neurons*2)
    # conv0_1 = residual_block(conv0_1, start_neurons*2)
    conv0_1 = LeakyReLU(alpha=0.1)(conv0_1)

    conv1_1 = Conv2DTranspose(start_neurons*4, (2,2), strides=(2, 2), padding="same")(conv2_0)
    pool0_1 = MaxPooling2D((2, 2))(conv0_1)
    conv1_1 = concatenate([conv1_1, conv1_0, pool0_1])
    conv1_1 = Conv2D(start_neurons*4, (1,1), activation=None, padding="same")(conv1_1)
    conv1_1 = residual_block(conv1_1, start_neurons*4)
    # conv1_1 = residual_block(conv1_1, start_neurons*4)
    conv1_1 = LeakyReLU(alpha=0.1)(conv1_1)

    conv2_1 = Conv2DTranspose(start_neurons*8, (2,2), strides=(2, 2), padding="same")(conv3_0)
    pool1_1 = MaxPooling2D((2, 2))(conv1_1)
    conv2_1 = concatenate([conv2_1, conv2_0, pool1_1])
    conv2_1 = Conv2D(start_neurons*8, (1,1), activation=None, padding="same")(conv2_1)
    conv2_1 = residual_block(conv2_1, start_neurons*8)
    # conv2_1 = residual_block(conv2_1, start_neurons*8)
    conv2_1 = LeakyReLU(alpha=0.1)(conv2_1)

    conv3_1 = Conv2DTranspose(start_neurons*16, (2,2), strides=(2, 2), padding="same")(conv4_0)
    pool2_1 = MaxPooling2D((2, 2))(conv2_1)
    conv3_1 = concatenate([conv3_1, conv3_0, pool2_1])
    conv3_1 = Conv2D(start_neurons*16, (1,1), activation=None, padding="same")(conv3_1)
    conv3_1 = residual_block(conv3_1, start_neurons*16)
    # conv3_1 = residual_block(conv3_1, start_neurons*16)
    conv3_1 = LeakyReLU(alpha=0.1)(conv3_1)

    conv4_1 = Conv2DTranspose(start_neurons*32, (2,2), strides=(2, 2), padding="same")(conv5_0)
    pool3_1 = MaxPooling2D((2, 2))(conv3_1)
    conv4_1 = concatenate([conv4_1, conv4_0, pool3_1])
    conv4_1 = Conv2D(start_neurons*32, (1,1), activation=None, padding="same")(conv4_1)
    conv4_1 = residual_block(conv4_1, start_neurons*32)
    # conv4_1 = residual_block(conv4_1, start_neurons*32)
    conv4_1 = LeakyReLU(alpha=0.1)(conv4_1)


    conv0_2 = Conv2DTranspose(start_neurons*2, (2,2), strides=(2, 2), padding="same")(conv1_1)
    conv0_2 = concatenate([conv0_2, conv0_1, conv0_0])
    conv0_2 = Conv2D(start_neurons*2, (1,1), activation=None, padding="same")(conv0_2)
    conv0_2 = residual_block(conv0_2, start_neurons*2)
    # conv0_2 = residual_block(conv0_2, start_neurons*2)
    conv0_2 = LeakyReLU(alpha=0.1)(conv0_2)

    conv1_2 = Conv2DTranspose(start_neurons*4, (2,2), strides=(2, 2), padding="same")(conv2_1)
    pool0_2 = MaxPooling2D((2, 2))(conv0_2)
    conv1_2 = concatenate([conv1_2, conv1_1, conv1_0, pool0_2])
    conv1_2 = Conv2D(start_neurons*4, (1,1), activation=None, padding="same")(conv1_2)
    conv1_2 = residual_block(conv1_2, start_neurons*4)
    # conv1_2 = residual_block(conv1_2, start_neurons*4)
    conv1_2 = LeakyReLU(alpha=0.1)(conv1_2)

    conv2_2 = Conv2DTranspose(start_neurons*8, (2,2), strides=(2, 2), padding="same")(conv3_1)
    pool1_2 = MaxPooling2D((2, 2))(conv1_2)
    conv2_2 = concatenate([conv2_2, conv2_1, conv2_0, pool1_2])
    conv2_2 = Conv2D(start_neurons*8, (1,1), activation=None, padding="same")(conv2_2)
    conv2_2 = residual_block(conv2_2, start_neurons*8)
    # conv2_2 = residual_block(conv2_2, start_neurons*8)
    conv2_2 = LeakyReLU(alpha=0.1)(conv2_2)

    conv3_2 = Conv2DTranspose(start_neurons*16, (2,2), strides=(2, 2), padding="same")(conv4_1)
    pool2_2 = MaxPooling2D((2, 2))(conv2_2)
    conv3_2 = concatenate([conv3_2, conv3_1, conv3_0, pool2_2])
    conv3_2 = Conv2D(start_neurons*16, (1,1), activation=None, padding="same")(conv3_2)
    conv3_2 = residual_block(conv3_2, start_neurons*16)
    # conv3_2 = residual_block(conv3_2, start_neurons*16)
    conv3_2 = LeakyReLU(alpha=0.1)(conv3_2)


    conv0_3 = Conv2DTranspose(start_neurons*2, (2,2), strides=(2, 2), padding="same")(conv1_2)
    conv0_3 = concatenate([conv0_3, conv0_2, conv0_1, conv0_0])
    conv0_3 = Conv2D(start_neurons*2, (1,1), activation=None, padding="same")(conv0_3)
    conv0_3 = residual_block(conv0_3, start_neurons*2)
    # conv0_3 = residual_block(conv0_3, start_neurons*2)
    conv0_3 = LeakyReLU(alpha=0.1)(conv0_3)

    conv1_3 = Conv2DTranspose(start_neurons*4, (2,2), strides=(2, 2), padding="same")(conv2_2)
    pool0_3 = MaxPooling2D((2, 2))(conv0_3)
    conv1_3 = concatenate([conv1_3, conv1_2, conv1_1, conv1_0, pool0_3])
    conv1_3 = Conv2D(start_neurons*4, (1,1), activation=None, padding="same")(conv1_3)
    conv1_3 = residual_block(conv1_3, start_neurons*4)
    # conv1_3 = residual_block(conv1_3, start_neurons*4)
    conv1_3 = LeakyReLU(alpha=0.1)(conv1_3)

    conv2_3 = Conv2DTranspose(start_neurons*8, (2,2), strides=(2, 2), padding="same")(conv3_2)
    pool1_3 = MaxPooling2D((2, 2))(conv1_3)
    conv2_3 = concatenate([conv2_3, conv2_2, conv2_1, conv2_0, pool1_3])
    conv2_3 = Conv2D(start_neurons*8, (1,1), activation=None, padding="same")(conv2_3)
    conv2_3 = residual_block(conv2_3, start_neurons*8)
    # conv2_3 = residual_block(conv2_3, start_neurons*8)
    conv2_3 = LeakyReLU(alpha=0.1)(conv2_3)

    conv0_4 = Conv2DTranspose(start_neurons*2, (2,2), strides=(2, 2), padding="same")(conv1_3)
    conv0_4 = concatenate([conv0_4, conv0_3, conv0_2, conv0_1, conv0_0])
    conv0_4 = Conv2D(start_neurons*2, (1,1), activation=None, padding="same")(conv0_4)
    conv0_4 = residual_block(conv0_4, start_neurons*2)
    # conv0_4 = residual_block(conv0_4, start_neurons*2)
    conv0_4 = LeakyReLU(alpha=0.1)(conv0_4)

    conv1_4 = Conv2DTranspose(start_neurons*4, (2,2), strides=(2, 2), padding="same")(conv2_3)
    pool0_4 = MaxPooling2D((2, 2))(conv0_4)
    conv1_4 = concatenate([conv1_4, conv1_3, conv1_2, conv1_1, conv1_0, pool0_4])
    conv1_4 = Conv2D(start_neurons*4, (1,1), activation=None, padding="same")(conv1_4)
    conv1_4 = residual_block(conv1_4, start_neurons*4)
    # conv1_4 = residual_block(conv1_4, start_neurons*4)
    conv1_4 = LeakyReLU(alpha=0.1)(conv1_4)

    conv0_5 = Conv2DTranspose(start_neurons*2, (2,2), strides=(2, 2), padding="same")(conv1_4)
    conv0_5 = concatenate([conv0_5, conv0_4, conv0_3, conv0_2, conv0_1, conv0_0])
    conv0_5 = Conv2D(start_neurons*2, (1,1), activation=None, padding="same")(conv0_5)
    conv0_5 = residual_block(conv0_5, start_neurons*2)
    conv0_5 = residual_block(conv0_5, start_neurons*2)
    conv0_5 = LeakyReLU(alpha=0.1)(conv0_5)

    conv0_5 = Dropout(dropout_rate/2)(conv0_5)
    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name="segmentation")(conv0_5)

    segmentation_model = Model(input, output)

    return segmentation_model


# Actual Function Calls Begin here

img_size = 512

def create_heatmap(predicted_mask, original_threshold):
    heatmap=predicted_mask
    heatmap[heatmap<0.0001] = 0
    heatmap_2 = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO) 
    heatmap_2[:, :, 0][heatmap==0] = 0
    heatmap_2[:, :, 1][heatmap==0] = 0
    heatmap_2[:, :, 2][heatmap==0] = 0
    
    return heatmap_2

def abnormality_model():
    seg_model = UppXceptionWithAttention(input_shape=(img_size,img_size,3))
    saved_weights_pth_unet = 'x.hdf5'

    seg_model.load_weights(saved_weights_pth_unet)
    
    return seg_model

def preprocessing(x):
    image2=cv2.resize(x, (img_size,img_size))
    if image2.max() < 256:
        image2 = image2.astype('float') / 255
    elif image2.max() < 4096:
        image2 = image2.astype('float') / 4095
    else:
        image2 = image2.astype('float') / image2.max()  
        
    image=np.expand_dims(image2, axis=0)
    return image


# def postprocessing(predicted_mask):
#     original_threshold=0.1
    
#     if np.max(predicted_mask[0,:,:,0])>original_threshold:
#         predicted_class=1
        
# #         predicted_mask=cv2.resize(predicted_mask, (img_size,img_size))
#         mask_test = np.reshape(predicted_mask[0,:,:,0],(img_size,img_size))
#         mask_test=np.where(mask_test > original_threshold,1,0)
#         mask_test = (mask_test*255.0).astype('uint8')
        
#     else:
#         predicted_class=0
#         mask_test=np.zeros(predicted_mask[0].shape, dtype='uint8')
#         mask_test = np.reshape(mask_test,(img_size,img_size))
#         mask_test[mask_test<original_threshold] = 0
        
#     return (original_threshold, np.max(predicted_mask[0,:,:,0]), predicted_class, mask_test)

def postprocessing(predicted_mask):
    original_threshold=0.1
    
    if np.max(predicted_mask[0,:,:,0])>original_threshold:
        predicted_class=1
        
        print (predicted_mask.shape)
        mask_test = np.reshape(predicted_mask[0,:,:,0],(512,512))
        mask_test=np.where(mask_test > original_threshold,1,0)
        mask_test=predicted_mask[0]
        
        mask_test=cv2.resize(mask_test, (768,768))
#         mask_test = np.reshape(mask_test,(768,768))
        
        heatmap_2=create_heatmap(mask_test, original_threshold)
    
        mask_test= np.where(mask_test>original_threshold,1,0)
        mask_test = (mask_test*255.0).astype('uint8')
        
    else:
        predicted_class=0
        
        mask_test=predicted_mask[0]
        mask_test=cv2.resize(mask_test, (768,768))
#         mask_test = np.reshape(mask_test,(768,768))
        heatmap_2=create_heatmap(mask_test, original_threshold)
        
        mask_test=np.zeros(predicted_mask[0].shape, dtype='uint8')
#         mask_test = np.reshape(mask_test,(768,768))
        mask_test=cv2.resize(mask_test, (768,768))
        mask_test[mask_test<original_threshold] = 0
        
#         heatmap_2=create_heatmap(mask_test, original_threshold)
        
    return (original_threshold, np.max(predicted_mask[0,:,:,0]), predicted_class, mask_test, heatmap_2)


def inference(x):
    predicted_mask=seg_model.predict(preprocessing(x),verbose=0)
    original_threshold, predicted_probability, predicted_class, predicted_mask = postprocessing(predicted_mask[0])
