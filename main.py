import os

## GAN Variants
from GAN import GAN
from CGAN import CGAN
from infoGAN import infoGAN
from ACGAN import ACGAN
from EBGAN import EBGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from DRAGAN import DRAGAN
from LSGAN import LSGAN
from BEGAN import BEGAN

## VAE Variants
from VAE import VAE
from CVAE import CVAE

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse
import random
import time

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'VAE', 'CVAE'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=1, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_CGAN_mnist',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results_CGAN_mnist',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs_CGAN_mnist',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def run_one_model(learning_rate_curr, disc_iter_curr, criterior_curr):
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    models = [GAN, CGAN, infoGAN, ACGAN, EBGAN, WGAN, WGAN_GP, DRAGAN,
              LSGAN, BEGAN, VAE, CVAE]

    tf.reset_default_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir,
                            learning_rate = learning_rate_curr,
                            disc_iter = disc_iter_curr,
                            criterior = criterior_curr
                           )
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        criterior_result, learning_rate_result, disc_iter_result = gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")
        
        return criterior_result, learning_rate_result, disc_iter_result

def grid_search(lr_list, disc_iter_list, criterior):
    lr_list_len = len(lr_list)
    disc_iter_list_len = len(disc_iter_list)

    for i in range(lr_list_len):
        for j in range(disc_iter_list_len):
            criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr = run_one_model(lr_list[i], disc_iter_list[j], criterior)

            if (i==0 and j == 0):
                best_criterior, best_learning_rate, best_disc_iter = criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr
            else:
                if (criterior <= 1):
                    if (criterior_result_curr > best_criterior):
                        best_criterior, best_learning_rate, best_disc_iter = criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr
                else:
                    if (criterior_result_curr < best_criterior):
                        best_criterior, best_learning_rate, best_disc_iter = criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr
    
    return best_criterior, best_learning_rate, best_disc_iter

def random_search(lr_list, disc_iter_list, criterior, try_times):
    lr_list_len = len(lr_list)
    disc_iter_list_len = len(disc_iter_list)
    
    for try_time in range(try_times):
        i = random.randint(0, lr_list_len - 1)
        j = random.randint(0, disc_iter_list_len - 1)

        criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr = run_one_model(lr_list[i], disc_iter_list[j], criterior)

        if (try_time == 0):
            best_criterior, best_learning_rate, best_disc_iter = criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr
        else:
            if (criterior <= 1):
                if (criterior_result_curr > best_criterior):
                    best_criterior, best_learning_rate, best_disc_iter = criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr
            else:
                if (criterior_result_curr < best_criterior):
                    best_criterior, best_learning_rate, best_disc_iter = criterior_result_curr, learning_rate_result_curr, disc_iter_result_curr
    return best_criterior, best_learning_rate, best_disc_iter

if __name__ == '__main__':
    start_time = time.time()
    best_cri, best_lr, best_d_it = grid_search([0.0002, 0.002, 0.02, 0.2], [1, 2], 2)
    print("criterior = %0.6f" % best_cri)
    print("best learning rate = %0.6f" % best_lr)
    print("best discriminator iteration time = %0.6f" % best_d_it)
    time_duration = time.time() - start_time
    print("total time = %0.2f" % time_duration)
