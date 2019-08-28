def CrowdNet():
    # Variable Input Size
    rows = 28
    cols = 28

    # Batch Normalisation option

    batch_norm = 0
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
