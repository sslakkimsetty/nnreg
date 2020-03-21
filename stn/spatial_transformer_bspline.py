class SpatialTransformerBspline(tf.keras.layers.Layer):

    def __init__(self, img_res=None, grid_res=None, out_dims=None, B=None):
        super(SpatialTransformerBspline, self).__init__()

        # !!! TODO support for multi channel featuremaps is missing!

        #### MAIN TRANSFORMER FUNCTION PART ####
        if not img_res:
            img_res = (40, 40)
        self.H, self.W = img_res

        if not grid_res:
            grid_res = (5.0, 5.0)
        sx, sy = grid_res
        self.sx, self.sy = tf.cast(sx, tf.float32), tf.cast(sy, tf.float32)

        if not out_dims:
            out_dims = img_res
        self.out_H, self.out_W = out_dims

        gx, gy = tf.math.ceil(self.W/sx), tf.math.ceil(self.H/sy)
        self.nx, self.ny = tf.cast(gx+3, tf.int32), tf.cast(gy+3, tf.int32)


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
        px, py = tf.floor(xt/sx), tf.floor(yt/sy)
        u = (xt/sx) - px
        v = (yt/sy) - py

        # Compute Bsplines
        Bu = self._piece_bsplines(u, px) ##
        Bu = tf.reshape(Bu, shape=(5,-1))
        self.Bu = tf.transpose(Bu)

        Bv = self._piece_bsplines(v, py) ##
        Bv = tf.reshape(Bv, shape=(5,-1))
        self.Bv = tf.transpose(Bv)

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
            theta = tf.reshape(theta, shape=[self.B,2,self.ny,self.nx])

        # # If grid res is not provided,
        # if not grid_res:
        #     grid_res = (5.0, 5.0)

        # sx, sy = grid_res
        # gx, gy = math.ceil(W/sx), math.ceil(H/sy)
        # nx, ny = int(gx+3), int(gy+3)
        # theta = tf.reshape(theta, shape=[B,2,ny,nx])
        # theta = tf.cast(theta, tf.float32)

        # Initialize out_dims to input dimensions if not provided
        # if out_dims:
        #     out_H, out_W = out_dims
        #     batch_grids = self._grid_generator(out_H, out_W, theta, grid_res) ##
        # else:
        #     batch_grids = self._grid_generator(H, W, theta, grid_res) ##

        # batch_grids = self._grid_generator(self.out_H, self.out_W, theta, grid_res) ##

        # theta = tf.reshape(theta, shape=[self.B,2,self.ny,self.nx])

        batch_grids = self._grid_generator(theta)

        # Extract source coordinates
        # batch_grids has shape (2,B,H,W)
        xs = batch_grids[0, :, :, :]
        ys = batch_grids[1, :, :, :]

        # Compile output feature map
        out_fmap = self._bilinear_sampler(input_fmap, xs, ys) ##
        return out_fmap


    def _grid_generator(self, theta=None):
        # # theta shape B, 2, nx, ny
        # B, _, nx, ny = theta.shape

        # sx, sy = grid_res

        # # Create meshgrid
        # x = tf.linspace(start=0.0, stop=W-1, num=W)
        # y = tf.linspace(start=0.0, stop=H-1, num=H)
        # xt, yt = tf.meshgrid(x, y)

        # # Calculate base indices and bspline inputs
        # px, py = tf.floor(xt/sx), tf.floor(yt/sy)
        # u = (xt/sx) - px
        # v = (yt/sy) - py

        # # Compute Bsplines
        # Bu = self._piece_bsplines(u, px) ##
        # Bu = tf.reshape(Bu, shape=(5,-1))
        # Bu = tf.transpose(Bu)

        # Bv = self._piece_bsplines(v, py) ##
        # Bv = tf.reshape(Bv, shape=(5,-1))
        # Bv = tf.transpose(Bv)

        # Compute offset for each pixel in input feature map
        delta = tf.map_fn(lambda x: self._delta_calculator(x[0], x[1], theta), ##
                          (self.Bu, self.Bv),
                          dtype=(tf.float32, tf.float32))
        delta = tf.stack(delta, axis=0)
        delta = tf.reshape(delta, (2,self.B,self.H,self.W))

        # xt = tf.expand_dims(xt, axis=0)
        # xt = tf.tile(xt, tf.stack([B,1,1]))

        # yt = tf.expand_dims(yt, axis=0)
        # yt = tf.tile(yt, tf.stack([B,1,1]))

        # batch_grids = tf.stack([xt, yt], axis=0)

        batch_grids = self.base_grid + delta
        return batch_grids


    def _delta_calculator(self, x, y, theta):

        x, px = x[:-1], x[-1]
        y, py = y[:-1], y[-1]

        # x = tf.cast(x, tf.float32)
        # y = tf.cast(y, tf.float32)

        _px = tf.linspace(start=px, stop=px+3, num=4)
        _py = tf.linspace(start=py, stop=py+3, num=4)
        pmeshx, pmeshy = tf.meshgrid(_px, _py)

        pmeshx = tf.cast(pmeshx, tf.int32)
        pmeshy = tf.cast(pmeshy, tf.int32)
        pmeshx, pmeshy = tf.transpose(pmeshx), tf.transpose(pmeshy)

        pmeshx = tf.expand_dims(pmeshx, axis=0)
        pmeshx = tf.tile(pmeshx, [self.B,1,1])

        pmeshy = tf.expand_dims(pmeshy, axis=0)
        pmeshy = tf.tile(pmeshy, [self.B,1,1])

        batch_idx = tf.range(0, self.B)
        batch_idx = tf.reshape(batch_idx, (self.B, 1, 1))

        b = tf.tile(batch_idx, [1, 4, 4])
        indices = tf.stack([b, pmeshy, pmeshx], axis=3)

        _theta_x = tf.gather_nd(theta[:,0], indices) # theta[:,0] is theta_x
        _theta_x = tf.cast(_theta_x, tf.float32)

        _theta_y = tf.gather_nd(theta[:,1], indices) # theta[:,1] is theta_y
        _theta_y = tf.cast(_theta_y, tf.float32)

        assert (len(x) == 4), "Control point span in x is not 4!"
        assert (len(y) == 4), "Control point span in y is not 4!"
        assert (_theta_x.shape == (self.B,4,4)), "Control point span in x is not (4,4)!"
        assert (_theta_x.shape == (self.B,4,4)), "Control point span in y is not (4,4)!"

        x = tf.reshape(x, (-1,1))
        y = tf.reshape(y, (1,-1))

        z = tf.matmul(x, y)
        z = tf.expand_dims(z, axis=0)
        z = tf.tile(z, tf.stack([self.B,1,1]))
        zx = tf.math.reduce_sum(z * _theta_x, axis=[1,2])
        zy = tf.math.reduce_sum(z * _theta_y, axis=[1,2])
        return (zx, zy)


    def _piece_bsplines(self, u, p):
        u2 = u ** 2
        u3 = u ** 3

        U0 = (-u3 + 3*u2 - 3*u + 1) / 6
        U1 = (3*u3 - 6*u2 + 4) / 6
        U2 = (-3*u3 + 3*u2 + 3*u + 1) / 6
        U3 = u3 / 6

        U = tf.stack([U0,U1,U2,U3,p], axis=0)
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

        batch_idx = tf.range(0, B)
        batch_idx = tf.reshape(batch_idx, (B, 1, 1))

        b = tf.tile(batch_idx, [1, H, W])
        indices = tf.stack([b, y, x], axis=3)
        return tf.gather_nd(img, indices)


    def call(self, input_fmap, theta=None):
        self.B = input_fmap.shape[0]
        out = self._transformer(input_fmap, theta)
        return out
