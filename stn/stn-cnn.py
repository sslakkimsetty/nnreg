%tensorflow_version 2.x
import tensorflow as tf
import numpy as np


################### LINEAR CLASS ###################
class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32, is_transformer_regressor=False):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype="float32"), trainable=True)
        if is_transformer_regressor:
            theta = tf.constant([ [1,0,0], [0,1,0] ])
            theta = tf.cast(theta, "float32")
            theta = tf.reshape(theta, shape=(units,))
            self.b = tf.Variable(initial_value=theta, trainable=True)
        else:
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(
                shape=(units,),
                dtype="float32"
                ), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b



################### LOCNET CLASS ###################
class LocNet(tf.keras.layers.Layer):

    def __init__(self):
        super(LocNet, self).__init__()

        self.conv1 = Conv2D(filters=8, kernel_size=7, activation="relu")
        self.maxpool1 = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
        self.maxpool2 = MaxPooling2D(pool_size=(2,2), strides=2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        return x



################### REGRESSOR CLASS ###################
class TransformationRegressor(tf.keras.layers.Layer):

    def __init__(self):
        super(TransformationRegressor, self).__init__()

        self.linear1 = Linear(units=32, input_dim=10*6*6)
        self.linear2 = Linear(units=6, input_dim=32, is_transformer_regressor=True)

    def call(self, inputs):
        x = self.linear1(inputs)
        x = tf.nn.tanh(x)
        x = self.linear2(x)
        x = tf.nn.tanh(x)
        return x



################### STN-CNN CLASS ###################
class SpatialTransformerCNN(tf.keras.models.Model):

    def __init__(self, drate=0.2):
        super(SpatialTransformerCNN, self).__init__()

        self.loc_net = LocNet()
        self.flatten = Flatten()
        self.transformer_regressor = TransformationRegressor()
        self.CNN = Sequential([
                               Conv2D(filters=10, kernel_size=(5,5), strides=1, activation="relu"),
                               MaxPooling2D(pool_size=(2,2)),
                               Conv2D(filters=20, kernel_size=(5,5), strides=1, activation="relu"),
                               MaxPooling2D(pool_size=(2,2)),
                               Flatten(),
                               Dense(units=1024, activation="relu"),
                               Dropout(rate=drate),
                               Dense(units=10, activation="softmax")
        ])

    def call(self, inputs):
        xs = self.loc_net(inputs)
        xs = self.flatten(xs)
        theta = self.transformer_regressor(xs)
        xt = transformer(inputs, theta, out_dims=[40,40])
        logits = self.CNN(xt)
        return(logits)

    def transform(self, inputs):
        inputs = inputs.astype("float32")
        xs = self.loc_net(inputs)
        xs = self.flatten(xs)
        theta = self.transformer_regressor(xs)
        xt = transformer(inputs, theta, out_dims=[40,40])
        return xt



###################

