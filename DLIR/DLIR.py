import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Reshape, Flatten, MaxPooling2D
from spatial_transformer.stn_bspline import SpatialTransformerBspline



class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32,
        is_transformer_regressor=False):
        super(Linear, self).__init__()

        if is_transformer_regressor:
            w_init = tf.zeros_initializer()
            theta = np.zeros(shape=(units,))
            theta = tf.cast(theta, "float32")
            self.b = tf.Variable(initial_value=theta, trainable=True)
        else:
            w_init = tf.random_normal_initializer()
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(
                shape=(units,),
                dtype="float32"
                ), trainable=True)

        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype="float32"), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b



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



class TransformationRegressor(tf.keras.layers.Layer):

    def __init__(self, theta_units=6, input_dim=None):
        super(TransformationRegressor, self).__init__()

        self.linear1 = Linear(units=theta_units*5, input_dim=input_dim)
        self.linear2 = Linear(units=theta_units, input_dim=theta_units*5,
                              is_transformer_regressor=True)

    def call(self, inputs):
        x = self.linear1(inputs)
        x = tf.nn.tanh(x)
        x = self.linear2(x)
        x = tf.nn.tanh(x)
        return x



class DLIR(tf.keras.models.Model):

    def __init__(self, drate=0.2, grid_res=None, img_res=None,
        input_dim=None):
        super(DLIR, self).__init__()
        self.grid_res = grid_res
        self.img_res = img_res
        self.drate = drate
        self.B = 1

        self.loc_net = LocNet()
        self.flatten = Flatten()
        self.transformer = SpatialTransformerBspline(img_res=self.img_res,
                                                     grid_res=self.grid_res,
                                                     out_dims=None,
                                                     B=self.B)

        sx, sy = self.grid_res
        H, W = self.img_res
        gx, gy = tf.math.ceil(W/sx), tf.math.ceil(H/sy)
        nx, ny = tf.cast(gx+3, tf.int32), tf.cast(gy+3, tf.int32)

        self.transformer_regressor = TransformationRegressor(theta_units=2*ny*nx, input_dim=input_dim)


    def call(self, inputs):
        xs = self.loc_net(inputs)
        xs = self.flatten(xs)
        xs = Dropout(rate=self.drate)(xs)
        theta = self.transformer_regressor(xs)
        xt = self.transformer(inputs, theta, self.B)
        return xt


    def transform(self, inputs):
        # inputs = inputs.astype("float32")
        inputs = tf.cast(inputs, "float32")
        xs = self.loc_net(inputs)
        xs = self.flatten(xs)
        theta = self.transformer_regressor(xs)
        xt = self.transformer(inputs, theta, self.B)
        return xt



