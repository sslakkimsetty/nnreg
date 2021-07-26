import tensorflow as tf
import numpy as np

class SpatialTransformerBspline(tf.keras.layers.Layer):

    def __init__(self, img_res=None, grid_res=None, out_dims=None, B=None):
        super(SpatialTransformerBspline, self).__init__()

        # !!! TODO support for multi channel featuremaps is missing!

        #### MAIN TRANSFORMER FUNCTION PART ####
        if not img_res:
            img_res = (100, 100)
        self.H, self.W = img_res

        if not out_dims:
            out_dims = img_res
        self.out_H, self.out_W = out_dims

        if not grid_res:
            grid_res = (tf.math.ceil(self.W/7), tf.math.ceil(self.H/7))

        nx, ny = grid_res
        self.nx, self.ny = tf.cast(nx, tf.int32), tf.cast(ny, tf.int32)

        gx, gy = self.nx-3, self.ny-3
        sx, sy = tf.cast(self.W/gx, tf.float32), tf.cast(self.W/gy, tf.float32)


        #### GRID GENERATOR PART ####
        # Create grid
        x = tf.linspace(start=0.0, stop=self.W-1, num=self.W)
        y = tf.linspace(start=0.0, stop=self.H-1, num=self.H)
        xt, yt = tf.meshgrid(x, y)

        xt = tf.expand_dims(xt, axis=0)
        xt = tf.tile(xt, tf.stack([B,1,1]))

        yt = tf.expand_dims(yt, axis=0)
        yt = tf.tile(yt, tf.stack([B,1,1]))

        self.base_grid = tf.stack([xt, yt], axis=0)

        # Calculate base indices and piece wise bspline function inputs
        self.px, self.py = tf.floor(xt/sx), tf.floor(yt/sy)
        u = (xt/sx) - self.px
        v = (yt/sy) - self.py

        self.px = tf.cast(self.px, tf.int32)
        self.py = tf.cast(self.py, tf.int32)

        # Compute Bsplines
        # Bu and Bv have shapes (B*H*W, 4)
        Bu = self._piece_bsplines(u) ##
        Bu = tf.reshape(Bu, shape=(4,-1))
        Bu = tf.transpose(Bu)
        Bu = tf.reshape(Bu, (-1,4,1))

        Bv = self._piece_bsplines(v) ##
        Bv = tf.reshape(Bv, shape=(4,-1))
        Bv = tf.transpose(Bv)
        Bv = tf.reshape(Bv, (-1,1,4))

        self.Buv = tf.matmul(Bu,Bv)
        self.Buv = tf.reshape(self.Buv, (B,self.H,self.W,4,4))

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
            theta = tf.zeros((self.B,2,self.ny,self.nx), tf.float32)
        else:
            try:
                theta = tf.reshape(theta, shape=[self.B,2,self.ny,self.nx])
            except:
                theta = tf.reshape(theta, shape=[-1,2,self.ny,self.nx])
                self.B = theta.shape[0]

        batch_grids, delta = self._grid_generator(theta)

        # Extract source coordinates
        # batch_grids has shape (2,B,H,W)
        xs = batch_grids[0, :, :, :]
        ys = batch_grids[1, :, :, :]

        # Compile output feature map
        out_fmap = self._bilinear_sampler(input_fmap, xs, ys) ##
        return out_fmap, delta


    def _grid_generator(self, theta=None):
        # theta shape B, 2, ny, nx
        theta_x = theta[:,0,:,:]
        theta_y = theta[:,1,:,:]

        px = self.px
        py = self.py

        batch_idx = tf.range(0, self.B)
        batch_idx = tf.reshape(batch_idx, (self.B, 1, 1))
        b = tf.tile(batch_idx, [1, self.H, self.W])

        theta_slices_x = self._delta_calculator(b, px, py, theta_x)
        theta_slices_y = self._delta_calculator(b, px, py, theta_y)

        theta_slices_x = tf.cast(theta_slices_x, tf.float32)
        theta_slices_y = tf.cast(theta_slices_y, tf.float32)

        delta_x = self.Buv[:self.B] * theta_slices_x
        delta_x = tf.reduce_sum(delta_x, axis=[-2,-1])
        delta_y = self.Buv[:self.B] * theta_slices_y
        delta_y = tf.reduce_sum(delta_y, axis=[-2,-1])

        delta = tf.stack([delta_x, delta_y], axis=0)

        batch_grids = self.base_grid[:,:self.B] + delta
        return batch_grids, delta


    def _delta_calculator(self, b, px, py, theta):
        px = px[0:self.B]
        py = py[0:self.B]

        t0 = self._compute_theta_slices(b, px, py, theta, 0)
        t1 = self._compute_theta_slices(b, px, py, theta, 1)
        t2 = self._compute_theta_slices(b, px, py, theta, 2)
        t3 = self._compute_theta_slices(b, px, py, theta, 3)

        t = tf.stack([t0,t1,t2,t3], axis=-1)
        return t


    def _compute_theta_slices(self, b, px, py, theta, i):
        ti0 = tf.gather_nd(theta, tf.stack([b,py+i,px+0],3))
        ti1 = tf.gather_nd(theta, tf.stack([b,py+i,px+1],3))
        ti2 = tf.gather_nd(theta, tf.stack([b,py+i,px+2],3))
        ti3 = tf.gather_nd(theta, tf.stack([b,py+i,px+3],3))

        ti = tf.stack([ti0,ti1,ti2,ti3], axis=-1)
        return ti


    def _piece_bsplines(self, u):
        u2 = u ** 2
        u3 = u ** 3

        U0 = (-u3 + 3*u2 - 3*u + 1) / 6
        U1 = (3*u3 - 6*u2 + 4) / 6
        U2 = (-3*u3 + 3*u2 + 3*u + 1) / 6
        U3 = u3 / 6

        U = tf.stack([U0,U1,U2,U3], axis=0)
        return U


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

        b = tf.tile(batch_idx, [1, H, W])
        indices = tf.stack([b, y, x], axis=3)
        return tf.gather_nd(img, indices)


    def call(self, input_fmap, theta=None, B=None):
        self.B = B
        out = self._transformer(input_fmap, theta)
        return out

