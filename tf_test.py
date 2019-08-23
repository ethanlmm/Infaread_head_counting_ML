import tensorflow as tf
from tensorflow import keras



def MCNN():
    inputs = keras.layers.Input(shape=(None, None, 3))

    conv_m = keras.layers.Conv2D(filters=20, kernel_size=(7, 7), strides=1, dilation_rate=1, padding='same',
                                 activation='relu')(inputs)
    conv_m = keras.layers.Conv2D(filters=40, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same',
                                 activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(filters=20, kernel_size=(5, 5), strides=1, dilation_rate=3, padding='same',
                                 ctivation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(filters=10, kernel_size=(5, 5), strides=1, dilation_rate=5, padding='same',
                                 activation='relu')(conv_m)

    conv_s = keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same',
                                 activation='relu')(inputs)
    conv_s = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=1, dilation_rate=2, padding='same',
                                 activation='relu')(conv_s)
    conv_s = keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=1, dilation_rate=3, padding='same',
                                 activation='relu')(conv_s)
    conv_s = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=1, dilation_rate=5, padding='same',
                                 activation='relu')(conv_s)

    conv_l = keras.layers.Conv2D(filters=16, kernel_size=(9, 9), strides=1, dilation_rate=1, padding='same',
                                 activation='relu')(inputs)
    conv_l = keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=1, dilation_rate=2, padding='same',
                                   activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=1, dilation_rate=3, padding='same',
                                 activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=1, dilation_rate=5, padding='same',
                                 activation='relu')(conv_l)

    conv_merge = keras.layers.Concatenate([conv_l, conv_m, conv_s])
    outs = keras.layers.Conv2D(1, (1, 1), padding='same')(conv_merge)
    m