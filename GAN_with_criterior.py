import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import criterior
tfgan = tf.contrib.gan

mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128
d_iter = 10

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

MODEL_GRAPH_DEF = 'classify_mnist_graph_def.pb'

fid_txt = open("fid.txt", "w")
wid_txt = open("wid.txt", "w")
kl_txt = open("kl.txt", "w")
js_txt = open("js.txt", "w")

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


G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

inc_score = criterior.mnist_score(images = G_sample, graph_def_filename= MODEL_GRAPH_DEF)
inc_score_new = criterior.mnist_score_new(images= G_sample, graph_def_filename= MODEL_GRAPH_DEF)
f_distance = criterior.mnist_frechet_distance(real_images = X,generated_images = G_sample, graph_def_filename = MODEL_GRAPH_DEF)
w_distance = criterior.mnist_frechet_distance_new(real_images = X,generated_images = G_sample, graph_def_filename = MODEL_GRAPH_DEF)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(100000):
    for _ in range(d_iter):
        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr, _,X_curr = sess.run(
            [D_solver, D_loss, clip_D, X],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr, G_sample_curr, inc_score_curr, inc_score_new_curr, f_distance_curr, w_distance_curr = sess.run(
        [G_solver, G_loss, G_sample, inc_score, inc_score_new, f_distance, w_distance],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        print("---f_distance_curr---")
        print(f_distance_curr)
        fid_txt.write(str(f_distance_curr) + "\n")

        print("---w_distance_curr---")
        print(w_distance_curr)
        wid_txt.write(str(w_distance_curr) + "\n")

        print("---inc_score_curr---")
        print(inc_score_curr)
        kl_txt.write(str(inc_score_curr) + "\n")

        print("---inc_score_new_curr---")
        print(inc_score_new_curr)
        js_txt.write(str(inc_score_new_curr) + "\n")


        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

fid_txt.close()
wid_txt.close()
kl_txt.close()
js_txt.close()
