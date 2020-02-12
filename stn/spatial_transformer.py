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
