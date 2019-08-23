import tensorflow as tf
from tensorflow import keras

def MCNN():
    inputs = keras.layers.Input(shape=(None, None, 3),name='global_input')
    #hybrid dilated convolution with mcnn
    conv_m = keras.layers.Conv2D(filters=20, kernel_size=(7, 7), strides=1, dilation_rate=1, padding='same',activation='relu')(inputs)
    conv_m = keras.layers.Conv2D(filters=40, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same',activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(filters=20, kernel_size=(5, 5), strides=1, dilation_rate=3, padding='same',activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(filters=10, kernel_size=(5, 5), strides=1, dilation_rate=5, padding='same',activation='relu')(conv_m)

    conv_s = keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same', activation='relu')(inputs)
    conv_s = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=1, dilation_rate=2, padding='same', activation='relu')(conv_s)
    conv_s = keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=1, dilation_rate=3, padding='same', activation='relu')(conv_s)
    conv_s = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=1, dilation_rate=5, padding='same', activation='relu')(conv_s)

    conv_l = keras.layers.Conv2D(filters=16, kernel_size=(9, 9), strides=1, dilation_rate=1, padding='same',activation='relu')(inputs)
    conv_l = keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=1, dilation_rate=2, padding='same',activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=1, dilation_rate=3, padding='same',activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=1, dilation_rate=5, padding='same', activation='relu')(conv_l)
    conv_merge = keras.layers.Concatenate(axis=3)([conv_l, conv_m, conv_s])
    outs = keras.layers.Conv2D(1, (1,1), padding='same')(conv_merge)
    mcnn=keras.Model(inputs,outs,name='mcnn')
    return mcnn

def maaae(y_true, y_pred):
    return abs(tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred))

def mssse(y_true, y_pred):
    return (tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred)) * (
                tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred))

#model.compile(optimizer='adam',loss='mse',metrics=[maaae, mssse])



def euclidean_distance_loss(y_true, y_pred):
    # Euclidean distance as a measure of loss (Loss function)
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))


def CrowdNet():
    # Variable Input Size
    rows = 28
    cols = 28

    # Batch Normalisation option

    batch_norm = True
    kernel = (3, 3)
    init = tf.keras.initializers.RandomNormal(stddev=0.01)
    model = tf.keras.models.Sequential(name='dilation')

    # custom VGG:

    if (batch_norm):
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, input_shape=(rows, cols, 1), activation='relu',padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())

    else:
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same',input_shape=(rows, cols, 1),kernel_initializer=init))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))

    # Conv2D
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init,
                                     padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init,
                                     padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init,
                                     padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init,
                                     padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init,
                                     padding='same'))
    model.add(
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(
        tf.keras.layers.Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=init, padding='same'))

    # sgd = SGD(lr=1e-7, decay=(5 * 1e-4), momentum=0.95)

    model.compile(optimizer='adam', loss=euclidean_distance_loss, metrics=['mse'])

    # model = init_weights_vgg(model)

    return model
