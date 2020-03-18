import tensorflow as tf
import numpy as np
import math



def spatial_transformer(input_fmap, theta=None, out_dims=None,
                        grid_res=None, **kwargs):
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

    # Input dimensions
    DIM = len(input_fmap.shape)

    B, H, W, C = input_fmap.shape
    if B == None:
        B = 1

    # Initialize theta to identity transformation if not provided
    if type(theta) == "NoneType":
        theta_x = np.zeros((1,ny,nx,1)) / 1.0
        theta_y = np.zeros((1,ny,nx,1)) / 1.0
        theta = tf.stack([theta_x, theta_y], axis=0)

    # If grid res is not provided,
    if not grid_res:
        grid_res = (5, 5)

    sx, sy = grid_res
    gx, gy = math.ceil(W/sx), math.ceil(H/sy)
    nx, ny = gx+3, gy+3
    theta = tf.reshape(theta, shape=[1,2,ny,nx])
    theta = tf.tile(theta, tf.stack([B,1,1,1]))
    theta = tf.cast(theta, tf.float32)

    # Initialize out_dims to input dimensions if not provided
    if out_dims:
        out_H, out_W = out_dims
        batch_grids = _grid_generator(out_H, out_W, theta, grid_res)
    else:
        batch_grids = _grid_generator(H, W, theta, grid_res)

    # Extract source coordinates
    xs = batch_grids[:, 0, :, :]
    ys = batch_grids[:, 1, :, :]

    # Compile output feature map
    out_fmap = _bilinear_sampler(input_fmap, xs, ys)

    return out_fmap


def _grid_generator(H, W, theta=None, grid_res=None):
    # theta shape B, 2, nx, ny
    batch_size, nx, ny, _ = theta.shape

    sx, sy = grid_res

    # Create meshgrid
    x = tf.linspace(start=0.0, stop=W-1, num=W)
    y = tf.linspace(start=0.0, stop=H-1, num=H)
    xt, yt = tf.meshgrid(x, y)

    # Calculate base indices and bspline inputs
    px, py = np.floor(xt/sx), np.floor(yt/sy)
    u = (xt/sx) - px
    v = (yt/sy) - py

    # Compute Bsplines
    Bu = _piece_bsplines(u, px)
    Bu = tf.reshape(Bu, shape=(5,-1))
    Bu = tf.transpose(Bu)

    Bv = _piece_bsplines(v, py)
    Bv = tf.reshape(Bv, shape=(5,-1))
    Bv = tf.transpose(Bv)

    # Compute theta patches
    theta_x = tf.expand_dims(theta[:,0,:,:], axis=3)
    theta_y = tf.expand_dims(theta[:,1,:,:], axis=3)

    theta_slices_x = tf.image.extract_patches(images=theta_x,
                                              sizes=[1,4,4,1],
                                              strides=[1,1,1,1],
                                              rates=[1,1,1,1],
                                              padding="VALID")
    theta_slices_x = tf.reshape(theta_slices_x, (batch_size,ny-3,nx-3,4,4))

    theta_slices_y = tf.image.extract_patches(images=theta_y,
                                              sizes=[1,4,4,1],
                                              strides=[1,1,1,1],
                                              rates=[1,1,1,1],
                                              padding="VALID")
    theta_slices_y = tf.reshape(theta_slices_y, (batch_size,ny-3,nx-3,4,4))

    # Compute offset for each pixel in input feature map
    # !!! TODO since _delta_calculator does not offer support for
    # batches, delta is a simple (2,H,W) shaped tensor
    # So, batchgrids are also without batches support
    delta = tf.map_fn(lambda x: _delta_calculator(x[0],x[1], theta_slices_x, theta_slices_y), (Bu,Bv),
                  dtype=(tf.float32, tf.float32))
    delta = tf.stack(delta, axis=0)
    delta = tf.reshape(delta, (2,H,W))

    batch_grids = tf.stack([xt, yt], axis=0)
    batch_grids += delta
    batch_grids = tf.expand_dims(batch_grids, axis=0)
    batch_grids = tf.tile(batch_grids, tf.stack([batch_size,1,1,1]))

    return batch_grids


def _delta_calculator(x, y, theta_slices_x, theta_slices_y):
    x, px = x[:-1], x[-1]
    y, py = y[:-1], y[-1]

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    px = tf.cast(px, tf.int32)
    py = tf.cast(py, tf.int32)

    # !!! TODO theta_slices_x has shape (B, H, W, 4, 4)
    # batches functionality is not present in this function
    # have to add that (for now B = 1) so this works
    _theta_x = theta_slices_x[0,px, py, :, :]
    _theta_x = tf.cast(_theta_x, tf.float32)

    _theta_y = theta_slices_y[0,px, py, :, :]
    _theta_y = tf.cast(_theta_y, tf.float32)

    assert (len(x) == 4), "Control point span in x is not 4!"
    assert (len(y) == 4), "Control point span in y is not 4!"
    assert (_theta_x.shape == (4,4)), "Control point span in x is not (4,4)!"
    assert (_theta_x.shape == (4,4)), "Control point span in y is not (4,4)!"

    x = tf.reshape(x, (-1,1))
    y = tf.reshape(y, (1,-1))

    z = tf.matmul(x, y)
    zx = np.sum(z * _theta_x)
    zy = np.sum(z * _theta_y)

    return (zx, zy)


def _piece_bsplines(u, p):
    u2 = u ** 2
    u3 = u ** 3

    U0 = (-u3 + 3*u2 - 3*u + 1) / 6
    U1 = (3*u3 - 6*u2 + 4) / 6
    U2 = (-3*u3 + 3*u2 + 3*u + 1) / 6
    U3 = u3 / 6

    U = tf.stack([U0,U1,U2,U3,p], axis=0)
    return U


def _bilinear_sampler(img, x, y):
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
    batch_size, H, W, C = img.shape

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
    Ia = _pixel_intensity(img, x0, y0) # bottom left
    Ib = _pixel_intensity(img, x0, y1) # top left
    Ic = _pixel_intensity(img, x1, y0) # bottom right
    Id = _pixel_intensity(img, x1, y1) # top right

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


def _pixel_intensity(img, x, y):
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

    shape = tf.shape(img)
    batch_size = shape[0]
    H = shape[1]
    W = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))

    b = tf.tile(batch_idx, [1, W, H])
    indices = tf.stack([b, y, x], axis=3)

    return tf.gather_nd(img, indices)

