import tensorflow as tf
from tensorflow import keras


def MCNN():
    inputs = keras.layers.Input(shape=(None, None, 1))
    conv_m = keras.layers.Conv2D(20, (7, 7), padding='same', activation='relu')(inputs)
    conv_m = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_m)
    conv_m = keras.layers.Conv2D(40, (5, 5), padding='same', activation='relu')(conv_m)
    conv_m = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_m)
    conv_m = keras.layers.Conv2D(20, (5, 5), padding='same', activation='relu')(conv_m)
    conv_m = keras.layers.Conv2D(10, (5, 5), padding='same', activation='relu')(conv_m)

    conv_s = keras.layers.Conv2D(24, (5, 5), padding='same', activation='relu')(inputs)
    conv_s = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_s)
    conv_s = keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu')(conv_s)
    conv_s = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_s)
    conv_s = keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu')(conv_s)
    conv_s = keras.layers.Conv2D(12, (3, 3), padding='same', activation='relu')(conv_s)

    conv_l = keras.layers.Conv2D(16, (9, 9), padding='same', activation='relu')(inputs)
    conv_l = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_l)

    conv_l = keras.layers.Conv2D(32, (7, 7), padding='same', activation='relu')(conv_l)
    conv_l = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_l)
    conv_l = keras.layers.Conv2D(16, (7, 7), padding='same', activation='relu')(conv_l)
    conv_l = keras.layers.Conv2D(8, (7, 7), padding='same', activation='relu')(conv_l)

    conv_merge = keras.layers.Concatenate(axis=3)([conv_m, conv_s, conv_l])
    result = keras.layers.Conv2D(1, (1, 1), padding='same')(conv_merge)

    model = keras.Model(inputs=inputs, outputs=result)
    return model


def maaae(y_true, y_pred):
    return abs(tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred))


def mssse(y_true, y_pred):
    return (tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred)) * (
                tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred))


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
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, input_shape=(rows, cols, 1), activation='relu',
                                         padding='same'))
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
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same',
                                         input_shape=(rows, cols, 1), kernel_initializer=init))
        model.add(
            tf.keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(
            tf.keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(
            tf.keras.layers.Conv2D(128, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(
            tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(
            tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(
            tf.keras.layers.Conv2D(256, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(strides=2))
        model.add(
            tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(
            tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))
        model.add(
            tf.keras.layers.Conv2D(512, kernel_size=kernel, activation='relu', padding='same', kernel_initializer=init))

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
