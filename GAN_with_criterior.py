import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import criterior
from network import generator_mlp_mnist, discriminator_mlp_mnist
from visualization import visualize
from utils import sample_z
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

X = tf.placeholder(tf.float32, shape=[None, X_dim])

z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_sample = generator_mlp_mnist(z)
D_real = discriminator_mlp_mnist(X)
D_fake = discriminator_mlp_mnist(G_sample)

inc_score = criterior.mnist_score(images = G_sample, graph_def_filename= MODEL_GRAPH_DEF)
inc_score_new = criterior.mnist_score_new(images= G_sample, graph_def_filename= MODEL_GRAPH_DEF)
f_distance = criterior.mnist_frechet_distance(real_images = X,generated_images = G_sample, graph_def_filename = MODEL_GRAPH_DEF)
w_distance = criterior.mnist_frechet_distance_new(real_images = X,generated_images = G_sample, graph_def_filename = MODEL_GRAPH_DEF)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss))

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

        _, D_loss_curr, X_curr = sess.run(
            [D_solver, D_loss, X],
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

            visualize(samples,i)

            i = i + 1

fid_txt.close()
wid_txt.close()
kl_txt.close()
js_txt.close()
