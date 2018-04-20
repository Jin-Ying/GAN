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
def run_one_gan(learning_rate_curr, disc_iter_curr, criterior_curr):
    # parse arguments
    args = parse_args()
    if args is None:
      exit()


    tf.reset_default_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None

        gan = GAN(sess,
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

def run_one_began(learning_rate_curr, gamma_curr, criterior_curr):
    # parse arguments
    args = parse_args()
    if args is None:
      exit()


    tf.reset_default_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None

        gan = BEGAN(sess,
                    epoch=args.epoch,
                    batch_size=args.batch_size,
                    z_dim=args.z_dim,
                    dataset_name=args.dataset,
                    checkpoint_dir=args.checkpoint_dir,
                    result_dir=args.result_dir,
                    log_dir=args.log_dir,
                    learning_rate = learning_rate_curr,
                    disc_iter = gamma_curr,
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

def run_one_model(parameter1, parameter2, criterior_curr):
    args = parse_args()

    if (args.gan_type == "GAN"):
        return run_one_gan(parameter1, parameter2, criterior_curr)
    elif (args.gan_type == "BEGAN")
        return run_one_began(parameter1, parameter2, criterior_curr)


def grid_search(list1, list2, criterior):
    list_len1 = len(list1)
    list_len2 = len(list2)

    for i in range(list_len1):
        for j in range(list_len2):
            criterior_result_curr, parameter1_result_curr, parameter2_result_curr = run_one_model(list1[i], list2[j], criterior)

            if (i==0 and j == 0):
                best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
            else:
                if (criterior <= 1):
                    if (criterior_result_curr > best_criterior):
                        best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
                else:
                    if (criterior_result_curr < best_criterior):
                        best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
    
    return best_criterior, best_parameter1, best_parameter2

def random_search(list1, list2, criterior, try_times):
    record_txt = open("random_search.txt","w")

    list_len1 = len(list1)
    list_len2 = len(list2)
    
    for try_time in range(try_times):
        i = random.randint(0, list_len1 - 1)
        j = random.randint(0, list_len2 - 1)

        criterior_result_curr, parameter1_result_curr, parameter2_result_curr = run_one_model(list1[i], list2[j], criterior)
        record_txt.write(str(parameter1_result_curr) + " " + str(parameter2_result_curr) + '\n')

        if (try_time == 0):
            best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
        else:
            if (criterior <= 1):
                if (criterior_result_curr > best_criterior):
                    best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
            else:
                if (criterior_result_curr < best_criterior):
                    best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr


    return best_criterior, best_parameter1, best_parameter2


def exploit_explore_search_1(list1, list2, criterior, try_times):
    record_txt = open("exploit_explore_1.txt","w")
    list_len1 = len(list1)
    list_len2 = len(list2)

    try_time =0

    while (try_time < try_times):
        i = random.randint(0, list_len1 - 1)
        j = random.randint(0, list_len2 - 1)

        criterior_result_curr, parameter1_result_curr, parameter2_result_curr = run_one_model(list1[i], list2[j], criterior)
        record_txt.write(str(parameter1_result_curr) + " " + str(parameter2_result_curr) + '\n')

        if (try_time == 0):
            best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
        else:
            if (criterior <= 1):
                if (criterior_result_curr > best_criterior):
                    best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
                    list1.append(best_parameter1 * 0.8)
                    list1.append(best_parameter1 * 1.2)
                    list2.append(best_parameter2 + 1)
                    list2.append(max(best_parameter2 - 1 ,1))
                    list_len1 = len(list1)
                    list_len2 = len(list2)
            else:
                if (criterior_result_curr < best_criterior):
                    best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
                    list1.append(best_parameter1 * 0.8)
                    list1.append(best_parameter1 * 1.2)
                    list2.append(best_parameter2 + 1)
                    list2.append(max(best_parameter2 - 1 ,1))
                    list_len1 = len(list1)
                    list_len2 = len(list2)
        try_time = try_time + 1

    return best_criterior, best_parameter1, best_parameter2

def exploit_explore_search_2(list1, list2, criterior, try_times):
    record_txt = open("exploit_explore_2.txt","w")
    list_len1 = len(list1)
    list_len2 = len(list2)

    try_time =0

    while (try_time < try_times):
        i = random.randint(0, list_len1 - 1)
        j = random.randint(0, list_len2 - 1)

        criterior_result_curr, parameter1_result_curr, parameter2_result_curr = run_one_model(list1[i], list2[j], criterior)
        record_txt.write(str(parameter1_result_curr) + " " + str(parameter2_result_curr) + '\n')

        if (try_time == 0):
            best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
        else:
            if (criterior <= 1):
                if (criterior_result_curr > best_criterior):
                    best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
                    list1.append(best_parameter1 * 0.8)
                    list1.append(best_parameter1 * 1.2)
                    list2.append(best_parameter2 * 0.8)
                    list2.append(best_parameter2 * 1.2)
                    list_len1 = len(list1)
                    list_len2 = len(list2)
            else:
                if (criterior_result_curr < best_criterior):
                    best_criterior, best_parameter1, best_parameter2 = criterior_result_curr, parameter1_result_curr, parameter2_result_curr
                    list1.append(best_parameter1 * 0.8)
                    list1.append(best_parameter1 * 1.2)
                    list2.append(best_parameter2 * 0.8)
                    list2.append(best_parameter2 * 1.2)
                    list_len1 = len(list1)
                    list_len2 = len(list2)
        try_time = try_time + 1

    return best_criterior, best_parameter1, best_parameter2

if __name__ == '__main__':
    start_time = time.time()
    best_cri, best_param1, best_param2 = grid_search([0.0002, 0.002, 0.02, 0.2], [1, 2], 2)
    print("criterior = %0.6f" % best_cri)
    print("best parameter1 = %0.6f" % best_param1)
    print("best parameter2 = %0.6f" % best_param2)
    time_duration = time.time() - start_time
    print("total time = %0.2f" % time_duration)
