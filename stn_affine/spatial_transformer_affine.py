import tensorflow as tf
import numpy as np

class SpatialTransformerAffine(tf.keras.layers.Layer):

    def __init__(self, img_res=None, out_dims=None, B=None):
        super(SpatialTransformerAffine, self).__init__()

        #### MAIN TRANSFORMER FUNCTION PART ####
        if not img_res:
            img_res = (100, 100)
        self.H, self.W = img_res

        if not out_dims:
            out_dims = img_res
        self.out_H, self.out_W = out_dims


        #### GRID GENERATOR PART ####
        # Create grid
        x = tf.linspace(start=0.0, stop=self.out_W-1, num=self.out_W)
        x = x / (self.out_W-1)
        y = tf.linspace(start=0.0, stop=self.out_H-1, num=self.out_H)
        y = y / (self.out_H-1)
        xt, yt = tf.meshgrid(x, y)

        xt = tf.expand_dims(xt, axis=0)
        xt = tf.tile(xt, tf.stack([B,1,1]))

        yt = tf.expand_dims(yt, axis=0)
        yt = tf.tile(yt, tf.stack([B,1,1]))

        self.base_grid = tf.stack([xt, yt], axis=0) # (2,B,H,W)
        #### ####


    def _transformer(self, input_fmap, theta=None):
        """
        Main transformer function that acts as a layer
        in a neural network. It does two things.
            1. Create a grid generator and transform the grid
                as per the transformation parameters
            2. Sample the input feature map using the transformed
                grid co-ordinates

        Args:
            input_fmap: the input feature map; shape=(B, H, W, C)
            theta:      transformation parameters; array of length
                        corresponding to a fn of grid_res
            out_dims:   dimensions of the output feature map (out_H, out_W)
                        if not provided, input dims are copied
            grid_res:   resolution of the control grid points (sx, sy)

        Returns:        output feature map of shape, out_dims
        """

        B, H, W, C = input_fmap.shape
        if B == None:
            self.B = 1

        # Initialize theta to identity transformation if not provided
        if type(theta) == type(None):
            theta = tf.zeros((self.B, 2, 3, C), tf.float32)
        else:
            try:
                theta = tf.reshape(theta, shape=[self.B,2,3,C])
            except:
                theta = tf.reshape(theta, shape=[-1,2,3,C])
                self.B = theta.shape[0]
                print(self.B)

        base_grid = tf.transpose(self.base_grid, [2,3,0,1])
        ones = np.ones((self.out_H, self.out_W, 1, self.B))
        base_grid = np.concatenate([base_grid, ones], axis=2)

        # !!! PROVISIONAL FIX ONLY
        theta = tf.squeeze(theta)
        # M1 = np.array([[1/(self.W-1), 2*np.pi/(self.W-1), 1/(self.W-1)],
        #                [2*np.pi/(self.H-1), 1/(self.H-1), 1/(self.H-1)]]).astype(np.float32)
        # M2 = np.array([[0.5/(self.W-1), -np.pi/(self.W-1), 0/(self.W-1)],
        #                [-np.pi/(self.H-1), 0.5/(self.H-1), 0/(self.H-1)]]).astype(np.float32)

        M1 = np.array([[1, np.pi, 0.2],
                       [np.pi, 1, 0.2]]).astype(np.float32)
        M2 = np.array([[0.5, -np.pi/2, -0.1],
                       [-np.pi/2, 0.5, -0.1]]).astype(np.float32)

        theta = tf.math.multiply(theta, M1) + M2

        batch_grids = tf.linalg.matmul(theta, base_grid)
        batch_grids = tf.transpose(batch_grids, [2,3,0,1])

        # Extract source coordinates
        # batch_grids has shape (2,B,H,W)
        xs = batch_grids[0, :, :, :]
        xs = xs * (self.W-1)
        ys = batch_grids[1, :, :, :]
        ys = ys * (self.H-1)

        # Compile output feature map
        out_fmap = self._bilinear_sampler(input_fmap, xs, ys) ##
        return out_fmap


    def _bilinear_sampler(self, img, x, y):
        """
        Implementation of garden-variety bilinear sampler,
        but samples for all batches and channels of the
        input feature map.

        Args:
            img:  the input feature map, expects shape
                of (B, H, W, C)
            x, y: the co-ordinates returned by grid generator
                in this context

        Returns: output feature map after sampling, returns
                in the shape (B, H, W, C)
        """
        B, H, W, C = img.shape

        # Define min and max of x and y coords
        zero = tf.zeros([], dtype=tf.int32)
        max_x = tf.cast(W-1, dtype=tf.int32)
        max_y = tf.cast(H-1, dtype=tf.int32)

        # Find corner coordinates
        x0 = tf.cast(tf.floor(x), dtype=tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # Clip corner coordinates to legal values
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # Get corner pixel values
        Ia = self._pixel_intensity(img, x0, y0) # bottom left ##
        Ib = self._pixel_intensity(img, x0, y1) # top left ##
        Ic = self._pixel_intensity(img, x1, y0) # bottom right ##
        Id = self._pixel_intensity(img, x1, y1) # top right ##

        # Define weights of corner coordinates using deltas
        # First recast corner coords as float32
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        # Weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # assert (wa + wb + wc + wd == 1.0), "Weights not equal to 1.0"

        # Add dimension for linear combination because
        # img = (B, H, W, C) and w = (B, H, W)
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # Linearly combine corner intensities with weights
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return out


    def _pixel_intensity(self, img, x, y):
        """
        Efficiently gather pixel intensities of transformed
        co-ordinates post sampling.
        Requires x and y to be of same shape

        Args:
            img:  the input feature map; shape = (B, H, W, C)
            x, y: co-ordinates (corner co-ordinates in bilinear
                sampling)

        Returns: the pixel intensities in the same shape and
                dimensions as x and y
        """

        B, H, W, C = img.shape
        if B == None:
            B = 1
            x = tf.expand_dims(x[0], axis=0)
            y = tf.expand_dims(y[0], axis=0)

        batch_idx = tf.range(0, B)
        batch_idx = tf.reshape(batch_idx, (B, 1, 1))

        b = tf.tile(batch_idx, [1, self.out_H, self.out_W])
        indices = tf.stack([b, y, x], axis=3)
        return tf.gather_nd(img, indices)


    def call(self, input_fmap, theta=None, B=None):
        self.B = B
        out = self._transformer(input_fmap, theta)
        return out



# 1st two paragraphs in text: The biotechnological problem is. ...
# the machine learning problem is ...
# the contributions is ...

# For slides as well


