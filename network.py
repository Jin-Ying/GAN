import tensorflow as tf

def generator_mlp_mnist(z):
    layer1 = tf.layers.dense(z, 128)
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.layers.dense(layer1, 128)
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.layers.dense(layer2, 784)
    G_prob = tf.nn.sigmoid(layer3)
    return G_prob


def discriminator_mlp_mnist(x):
    layer1 = tf.layers.dense(x, 128)
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.layers.dense(layer1,128)
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.layers.dense(layer2, 1)
    result = tf.nn.sigmoid(layer3)
    return result



