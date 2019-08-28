import tensorflow as tf
from tensorflow import keras


def MCNN():
    inputs = keras.layers.Input(shape=(None, None, 1))
    conv_m = keras.layers.Conv2D(20, (3, 3), padding='same', dilation_rate=1,activation='relu')(inputs)
    conv_m = keras.layers.Conv2D(40, (3, 3), padding='same', dilation_rate=1,activation='relu')(conv_m)
    conv_m = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_m)
    conv_m = keras.layers.Conv2D(80, (3, 3), padding='same',dilation_rate=1, activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(160, (3, 3), padding='same', dilation_rate=1, activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(80, (3, 3), padding='same', dilation_rate=1, activation='relu')(conv_m)
    conv_m = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_m)
    conv_m = keras.layers.Conv2D(40, (3, 3), padding='same',dilation_rate=1, activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(20, (5, 5), padding='same',dilation_rate=1, activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(10, (5, 5), padding='same',dilation_rate=1,  activation='relu')(conv_m)

    conv_s = keras.layers.Conv2D(12, (3, 3), padding='same', dilation_rate=1,activation='relu')(inputs)
    conv_s = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_s)
    conv_s = keras.layers.Conv2D(24, (3, 3), padding='same', dilation_rate=1, activation='relu')(conv_s)
    conv_s = keras.layers.Conv2D(48, (3, 3), padding='same', dilation_rate=1, activation='relu')(conv_s)
    conv_s = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_s)
    conv_s = keras.layers.Conv2D(24, (3, 3), padding='same', dilation_rate=1, activation='relu')(conv_s)
    conv_s = keras.layers.Conv2D(12, (3, 3), padding='same', dilation_rate=1, activation='relu')(conv_s)

    conv_l = keras.layers.Conv2D(32, (3, 3), padding='same',  dilation_rate=1,activation='relu')(inputs)
    conv_l = keras.layers.Conv2D(64, (3, 3), padding='same',  dilation_rate=1,activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(128, (3, 3), padding='same',  dilation_rate=1,activation='relu')(conv_l)
    conv_l = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_l)
    conv_l = keras.layers.Conv2D(256, (3, 3), padding='same',  dilation_rate=1,activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(512, (3, 3), padding='same',  dilation_rate=1,activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(256, (3, 3), padding='same',  dilation_rate=1, activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(128, (3, 3), padding='same',  dilation_rate=1,activation='relu')(conv_l)
    conv_l = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_l)
    conv_l = keras.layers.Conv2D(64, (3, 3), padding='same',  dilation_rate=1, activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(32, (3, 3), padding='same',  dilation_rate=1,activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(16, (3, 3), padding='same',  dilation_rate=1,activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(8, (3, 3), padding='same',  dilation_rate=1, activation='relu')(conv_l)

    conv_merge = keras.layers.Concatenate(axis=3)([conv_m, conv_s, conv_l])
    result = keras.layers.Conv2D(1, (1, 1), padding='same')(conv_merge)

    model = keras.Model(inputs=inputs, outputs=result)
    return model

def CrowdNet():
    inputs = keras.layers.Input(shape=(None, None, 1))
    inputs= keras.layers.Conv2D(64, 3, padding='same', dilation_rate=1,activation='relu')(inputs)
    inputs= keras.layers.Conv2D(64, 3, padding='same', dilation_rate=1,activation='relu')(inputs)
    inputs= keras.layers.MaxPooling2D(pool_size=(2, 2))(inputs)

    inputs = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=1, activation='relu')(inputs)
    inputs = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=1, activation='relu')(inputs)
    inputs = keras.layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    inputs = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(inputs)
    inputs = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(inputs)




def CSR_Net():
    inputs = keras.layers.Input(shape=(None, None, 1))
    vgg= keras.layers.Conv2D(64, 3, padding='same', dilation_rate=1, activation='relu')(inputs)
    vgg = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    vgg = keras.layers.MaxPooling2D(pool_size=(2, 2))(vgg)
    vgg = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    vgg = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    vgg = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    vgg = keras.layers.MaxPooling2D(pool_size=(2, 2))(vgg)
    vgg = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    vgg = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    vgg = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=1, activation='relu')(vgg)

    block1 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    block1 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=1, activation='relu')(block1)
    block1 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=1, activation='relu')(block1)
    block1 = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(block1)
    block1 = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=1, activation='relu')(block1)
    block1 = keras.layers.Conv2D(64, 3, padding='same', dilation_rate=1, activation='relu')(block1)


    block2 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(vgg)
    block2 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(block2)
    block2 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(block2)
    block2 = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=2, activation='relu')(block2)
    block2 = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=2, activation='relu')(block2)
    block2 = keras.layers.Conv2D(64, 3, padding='same', dilation_rate=2, activation='relu')(block2)

    block3 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=1, activation='relu')(vgg)
    block3 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(block3)
    block3 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(block3)
    block3 = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=4, activation='relu')(block3)
    block3 = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=4, activation='relu')(block3)
    block3 = keras.layers.Conv2D(64, 3, padding='same', dilation_rate=4, activation='relu')(block3)

    block4 = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=4, activation='relu')(vgg)
    block4 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=4, activation='relu')(block4)
    block4 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=4, activation='relu')(block4)
    block4 = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=4, activation='relu')(block4)
    block4 = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=4, activation='relu')(block4)
    block4 = keras.layers.Conv2D(64, 3, padding='same', dilation_rate=4, activation='relu')(block4)


    concat = keras.layers.Concatenate(axis=3)([block1,block2,block3])
    end = keras.layers.Conv2D(1,1,dilation_rate=1,activation='softmax')(concat)
    model = keras.Model(inputs=inputs,outputs=end)
    return model


def lmm():
    inputs = keras.layers.Input(shape=(None, None, 1))
    block1 = keras.layers.Conv2D(128, 5, padding='same',  dilation_rate=1,activation='relu')(inputs)
    block1 = keras.layers.Conv2D(256, 5, padding='same',  dilation_rate=1, activation='relu')(block1)

    block2 = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=2, activation='relu')(block1)
    block2 = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=5, activation='relu')(block2)

    block3 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(block2)
    block3 = keras.layers.Conv2D(512, 3, padding='same', dilation_rate=5, activation='relu')(block3)

    concat = keras.layers.Concatenate(axis=3)([block1, block2,block3])

    end = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)
    end = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(end)
    end = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(end)
    end = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=1, activation='relu')(end)
    end = keras.layers.MaxPooling2D(pool_size=(2, 2))(end)
    end = keras.layers.Conv2D(128, 3, padding='same', dilation_rate=1, activation='relu')(end)
    end = keras.layers.Conv2D(64, 3, padding='same', dilation_rate=1, activation='relu')(end)
    end = keras.layers.Conv2D(1, 1, padding='same', dilation_rate=1, activation='relu')(end)
    model=keras.Model(inputs=inputs,outputs=end)
    return model


def mae(y_true, y_pred):
    return abs(tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred))

def mse(y_true, y_pred):
    return (tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred)* (
                tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred)))




def euclidean_distance_loss(y_true, y_pred):
    # Euclidean distance as a measure of loss (Loss function)
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))


