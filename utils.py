import numpy as np
import tensorflow as tf

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

