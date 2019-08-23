import tensorflow as tf


class Mcnn(tf.keras.Model):
    def __init__(self):
        super(Mcnn, self).__init__()

        self.conv_m = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=20, kernel_size=(7, 7), strides=1,
                                                                  padding='same',
                                                                  activation='relu',),
                                           tf.keras.layers.Conv2D(filters=40, kernel_size=(5, 5), strides=1,
                                                                  padding='same',
                                                                  activation='relu'),
                                           tf.keras.layers.Conv2D(filters=20, kernel_size=(5, 5), strides=1,
                                                                  padding='same',
                                                                  activation='relu'),
                                           tf.keras.layers.Conv2D(filters=10, kernel_size=(5, 5), strides=1,
                                                                  padding='same',
                                                                  activation='relu')])
        self.conv_s = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=1,
                                                                  padding='same',
                                                                  activation='relu', ),
                                           tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=1,
                                                                  padding='same',
                                                                  activation='relu'),
                                           tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=1,
                                                                  padding='same',
                                                                  activation='relu'),
                                           tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=1,
                                                                  padding='same',
                                                                  activation='relu')])
        self.conv_l = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 9), strides=1,
                                                                  padding='same',
                                                                  activation='relu', ),
                                           tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=1,
                                                                  padding='same',
                                                                  activation='relu'),
                                           tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=1,
                                                                  padding='same',
                                                                  activation='relu'),
                                           tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=1,
                                                                  padding='same',
                                                                  activation='relu')])
        self.conv_merge = tf.keras.layers.Concatenate(axis=3)
        self.out=tf.keras.layers.Conv2D(1, (1, 1), padding='same')
        self.flatten=tf.keras.layers.Flatten()
        self.dense1=tf.keras.layers.Dense(32, activation='relu')
        self.dense2=tf.keras.layers.Dense(10, activation='softmax')


    def call(self, inputs, training=False, mask=None):
        convs=self.conv_s(inputs)
        convl=self.conv_l(inputs)
        convm=self.conv_m(inputs)
        x=self.conv_merge([convl,convm,convs])
        x=self.out(x)
        x=self.flatten(x)
        x=self.dense1(x)
        x=self.dense2(x)
        return x


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
                                         input_shape=(rows, cols, 1),
                                         kernel_initializer=init))
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
