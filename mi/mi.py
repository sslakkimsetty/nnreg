import tensorflow as tf
import numpy as np

def sampleCoords(dims, n):
    # dims are (H,W)
    H, W = dims

    ix = np.random.choice(W, size=n)
    iy = np.random.choice(H, size=n)

    coords = (iy, ix)
    return coords


def Gphi(z, phi, _type="marginal"):
    # if type is "joint", z is expected in nx2 shape
    # n = len(z)

    if _type == "marginal":
        phi_det = phi
        C = (-1/2) * ((z**2) / phi)
        k = 1
    else:
        phi_det = tf.linalg.det(phi)
        _A = tf.linalg.inv(phi)
        _B = tf.matmul(z, _A)
        _D = _B * z
        C = (-1/2) * (tf.reduce_sum(_D, axis=1))
        k = 2

    A = (2 * np.pi) ** (-k/2)
    B = phi_det ** (-1/2)
    return A * B * tf.exp(C)


def construct_z(img, c):
    cix, ciy, cjx, cjy = c
    n = len(cix)

    zi = tf.gather_nd(img, np.vstack([ciy, cix]).T)
    zj = tf.gather_nd(img, np.vstack([cjy, cjx]).T)

    zi = tf.reshape(tf.tile(zi, [n]), (n,-1))
    zj = tf.reshape(tf.tile(zj, [n]), (-1,n))
    zj = tf.transpose(zj)

    z = zi - zj
    return tf.reshape(z, (-1,))


def _entropy(z, n, _type="marginal", phi=0.1):
    g = Gphi(z, phi=phi, _type=_type)
    out = tf.reshape(g, (n,-1))
    out = (1/n) * tf.reduce_sum(out, axis=1)
    out = tf.math.log(out)
    out = -(1/n) * tf.reduce_sum(out)
    return out


def _compute_scale(z):
    return np.sqrt(np.var(z))


def mi(u, v, n=100):
    u = tf.squeeze(u)
    v = tf.squeeze(v)
    H, W = u.shape
    dims = (H, W)

    # Sample coordinates for sample B
    ciy, cix = sampleCoords(dims, n=n)

    # Sample coordinates for sample A
    cjy, cjx = sampleCoords(dims, n=n)

    c = (cix, ciy, cjx, cjy)

    # Construct z for u and v
    uz = construct_z(u, c)
    vz = construct_z(v, c)

    n = len(cix)

    phi = np.average([_compute_scale(uz),
                      _compute_scale(vz)])
    phi = np.sqrt(0.1)
    sigma = np.eye(2) * phi

    # Entropy for u and v
    hu = _entropy(uz, n, phi=phi)
    hv = _entropy(vz, n, phi=phi)

    # print("hu:", hu.numpy())
    # print("hv:", hv.numpy())

    # Joint entropy
    uvz = tf.stack([uz, vz])
    uvz = tf.transpose(uvz)
    huv = _entropy(uvz, n, _type="joint", phi=tf.eye(2,2)*phi)
    # print("huv:", huv.numpy())

    _mi = hu + hv - huv
    return _mi


