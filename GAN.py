import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
tfgan = tf.contrib.gan

mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out

MODEL_GRAPH_DEF = 'classify_mnist_graph_def.pb'
INPUT_TENSOR = 'inputs:0'
OUTPUT_TENSOR = 'logits:0'

def _graph_def_from_par_or_disk(filename):
    if filename is None:
      return tfgan.eval.get_graph_def_from_resource(MODEL_GRAPH_DEF)
    else:
      return tfgan.eval.get_graph_def_from_disk(filename)


def mnist_frechet_distance(real_images, generated_images,
                           graph_def_filename=None, input_tensor=INPUT_TENSOR,
                           output_tensor=OUTPUT_TENSOR, num_batches=1):
  # """Frechet distance between real and generated images.
    graph_def = _graph_def_from_par_or_disk(graph_def_filename)
    mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(
        x, graph_def, input_tensor, output_tensor)

    frechet_distance = tfgan.eval.frechet_classifier_distance(
      tf.reshape(real_images,[-1,28,28,1]), tf.reshape(generated_images,[-1,28,28,1]), mnist_classifier_fn, num_batches)
    return frechet_distance

G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

f_distance = mnist_frechet_distance(real_images = X,generated_images = G_sample, graph_def_filename = MODEL_GRAPH_DEF)
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    for _ in range(10):
        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr, _,X_curr = sess.run(
            [D_solver, D_loss, clip_D, X],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr, G_sample_curr, f_distance_curr = sess.run(
        [G_solver, G_loss, G_sample, f_distance],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))
        print("---X---")
        print(X_mb)
        print("---G_sample_curr---")
        print(G_sample_curr)
        print("---f_distance_curr---")
        print(f_distance_curr)

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)