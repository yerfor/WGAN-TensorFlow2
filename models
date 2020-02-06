import tensorflow as tf


class Generator(tf.keras.models.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.upsample_1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4,
                                                          strides=1, padding='valid',
                                                          use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.ReLU()

        self.upsample_2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same',
                                                          use_bias=False)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.ReLU()

        self.upsample_3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same',
                                                          use_bias=False)
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.activation_3 = tf.keras.layers.ReLU()

        self.upsample_4 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same',
                                                          use_bias=False)
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.activation_4 = tf.keras.layers.ReLU()

        self.upsample_5 = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same')
        self.activation_5 = tf.keras.layers.Activation(activation='tanh')

    def call(self, inputs, training=None, mask=None):
        # (b, 1, 1, 128) -> (b, 4, 4, 512)
        x = self.upsample_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.activation_1(x)

        # (128, 4, 4, 512) -> (128, 8, 8, 256)
        x = self.upsample_2(x)
        x = self.bn_2(x, training=training)
        x = self.activation_2(x)

        # (128, 8, 8, 256) -> (128, 16, 16, 128)
        x = self.upsample_3(x)
        x = self.bn_3(x, training=training)
        x = self.activation_3(x)

        # (128, 16, 16, 128) -> (128, 32, 32, 64)
        x = self.upsample_4(x)
        x = self.bn_4(x, training=training)
        x = self.activation_4(x)

        # (128, 32, 32, 64) -> (128, 64, 64, 3)
        x = self.upsample_5(x)
        x = self.activation_5(x)

        return x


class Discriminator(tf.keras.models.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                             padding='same')
        self.activation_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.activation_3 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.activation_4 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv_5 = tf.keras.layers.Conv2D(filters=1, kernel_size=4, padding='valid')

    def call(self, inputs, training=None, mask=None):
        # (b,64,64,3) -> (b,32,32,64)
        x = self.conv_1(inputs)
        x = self.activation_1(x)

        # (b,32,32,64) -> (b,16,16,128)
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.activation_2(x)

        # (b,16,16,128) -> (b,8,8,256)
        x = self.conv_3(x)
        x = self.bn_3(x, training=training)
        x = self.activation_3(x)

        # (b,8,8,256) -> (b,4,4,512)
        x = self.conv_4(x)
        x = self.bn_4(x, training=training)
        x = self.activation_4(x)

        # [b,4,4,512] -> [b,1,1,1]
        x = self.conv_5(x)

        return x


if __name__ == '__main__':
    input_shape = (128, 1, 1, 128)
    g = Generator()
    d = Discriminator()
    z = tf.random.normal(input_shape)

    x = g(z)
    print(x.shape)
    y = d(x)
    print(y.shape)
