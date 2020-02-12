import tensorflow as tf


def _grid_generator(H, W, theta):
    batch_size = theta.shape[0]

    # Create meshgrid
    x = tf.linspace(start=-1.0, stop=1.0, num=W)
    y = tf.linspace(start=-1.0, stop=1.0, num=H)
    xt, yt = tf.meshgrid(x, y)

    # Make xt, yt, ones as flattened tensors
    xtf = tf.reshape(xt, shape=[-1])
    ytf = tf.reshape(yt, shape=[-1])
    ones = tf.ones_like(xtf)

    # Create sampling grid for one batch and then broadcast it to batch_size
    sampling_grid = tf.stack([xtf, ytf, ones], axis=0)
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([batch_size, 1, 1]))

    # Cast theta and sampling_grid as float32 - matmul requires it
    theta = tf.cast(theta, "float32")
    sampling_grid = tf.cast(sampling_grid, "float32")

    # Transform the sampling_grid using matmul and reshape
    batch_grids = tf.matmul(theta, sampling_grid)
    batch_grids = tf.reshape(batch_grids, [batch_size, 2, H, W])

    return batch_grids


def _bilinear_sampler(img, x, y):
    H = tf.cast(tf.shape(img)[1], "float32")
    W = tf.cast(tf.shape(img)[2], "float32")

    # De-normalize x, y to W, H
    x = tf.cast(x, "float32")
    y = tf.cast(y, "float32")
    x = (x/2 + 0.5) * W
    y = (y/2 + 0.5) * H

    # Define min and max of x and y coords
    zero = tf.zeros([], dtype="int32")
    max_x = tf.cast(W-1, dtype="int32")
    max_y = tf.cast(H-1, dtype="int32")

    # Find corner coordinates
    x0 = tf.cast(tf.floor(x), "int32")
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), "int32")
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
    x0 = tf.cast(x0, "float32")
    x1 = tf.cast(x1, "float32")
    y0 = tf.cast(y0, "float32")
    y1 = tf.cast(y1, "float32")

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
    shape = tf.shape(img)
    batch_size = shape[0]
    H = shape[1]
    W = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))

    b = tf.tile(batch_idx, (1, W, H))
    indicies = tf.stack([b, y, x], axis=3)

    return tf.gather_nd(img, indicies)

