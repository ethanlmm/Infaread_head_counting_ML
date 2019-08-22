import tensorflow as tf



def maaae(y_true, y_pred):
    return abs(tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred))
def mssse(y_true, y_pred):
    return (tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred)) * (tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred))

def MCNN():
    input_layer = tf.keras.layers.Input(shape=(None, None, 1))
    conv_m = tf.keras.Sequential([input_layer,
                                  tf.keras.layers.Conv2D(filters=20, kernel_size=(7, 7), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.MaxPool2D(pool_size=2),
                                  tf.keras.layers.Conv2D(filters=40, kernel_size=(5, 5), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.MaxPool2D(pool_size=2),
                                  tf.keras.layers.Conv2D(filters=20, kernel_size=(5, 5), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.Conv2D(filters=10, kernel_size=(5, 5), strides=1, padding='same',
                                                         activation='relu')])

    conv_s = tf.keras.Sequential([input_layer,
                                  tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.MaxPool2D(pool_size=2),
                                  tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.MaxPool2D(pool_size=2),
                                  tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=1, padding='same',
                                                         activation='relu')])

    conv_l = tf.keras.Sequential([input_layer,
                                  tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 9), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.MaxPool2D(pool_size=2),
                                  tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.MaxPool2D(pool_size=2),
                                  tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=1, padding='same',
                                                         activation='relu'),
                                  tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=1, padding='same',
                                                         activation='relu')])

    conv_merge = tf.keras.layers.Concatenate(axis=3)([conv_l.output, conv_m.output, conv_s.output])
    out = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(conv_merge)
    model = tf.keras.Model(input_layer, out, name='mcnn')
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[maaae, mssse])
    return  model






def euclidean_distance_loss(y_true, y_pred):
    # Euclidean distance as a measure of loss (Loss function)
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))

def CrowdNet():
    # Variable Input Size
    rows = None
    cols = None

    # Batch Normalisation option

    batch_norm = 0
    kernel = (3, 3)
    init = tf.keras.initializers.RandomNormal(stddev=0.01)
    model = tf.keras.models.Sequential(name='dilation')

    # custom VGG:

    if (batch_norm):
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, input_shape=(rows, cols, 3), activation='relu', padding='same'))
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
        model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel, activation='relu', padding='same', input_shape=(rows, cols, 3),
                         kernel_initializer=init))
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
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same'))
    model.add(tf.keras.layers.Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=init, padding='same'))

    #sgd = SGD(lr=1e-7, decay=(5 * 1e-4), momentum=0.95)

    model.compile(optimizer='adam', loss=euclidean_distance_loss, metrics=['mse'])

    #model = init_weights_vgg(model)

    return model





