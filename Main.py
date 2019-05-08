import matplotlib.pyplot as plt
import scipy.misc
from random import randint
import numpy

from Parameters import data_params
from Parameters import model_params
from Parameters import hyper_params
from BeesDataGenerator import BeesDataGenerator
from Unet import Unet

from Tools import generate_data
from Tools import save_images
from Tools import segment_images
from Tools import count_bees_on_segmented_images
from Tools import print_parameters
from Tools import load_unet
from FileTools import empty_or_create_directory

import argparse

#----------------------------
# main routine
#----------------------------

parser = argparse.ArgumentParser(description='Bee Segmentation.')
parser.add_argument('-g', '--generate',
                    help='with this option program will only generate data for the 1st iteration of unsupervised training',
                    dest='generatedata', action='store_true')
parser.add_argument('-t', '--train',
                    help='with this option the program will train Unet model and store it',
                    dest='train', action='store_true')
parser.add_argument('-s', '--segment',
                    help='with this option the program will segment images',
                    dest='segment', action='store_true')
parser.add_argument('-c', '--count',
                    help='count segmented images vs manually labelled validation images',
                    dest='count', action='store_true')
args = parser.parse_args()

# print current parameters
print_parameters(data_params, model_params, hyper_params)

# paths to files with numpy data 
data_x_file_name = 'data/train_r{}/bees_x_r{}.npy'.format(hyper_params["bee_radius"], hyper_params["bee_radius"])
data_y_file_name = 'data/train_r{}/bees_y_r{}.npy'.format(hyper_params["bee_radius"], hyper_params["bee_radius"])
generated_images_dir = 'data/train_r{}/sample_images'.format(hyper_params["bee_radius"])
# path to trained model
model_path = 'data/train_r{}/model_weights_r{}_w{}.h5'.format(hyper_params["bee_radius"], hyper_params["bee_radius"], hyper_params["window_size"])

if args.generatedata:

    # step 1 - generate
    generate_data(data_params["labelled_train_db"],
                  data_params["labelled_train_dir"],
                  data_x_file_name,
                  data_y_file_name,
                  height = data_params["height"],
                  width = data_params["width"],
                  bee_radius = hyper_params["bee_radius"],
                  min_bee_prob = hyper_params["min_bee_prob"],
                  max_bee_prob = hyper_params["max_bee_prob"],
                  max_training_images_count = hyper_params["max_training_images_count"])

    # verify data visually
    data_x = numpy.load(data_x_file_name)
    data_y = numpy.load(data_y_file_name)
    save_images(5, data_x, data_y, generated_images_dir)

if args.train:

    # step 2 - train    
    unet = load_unet(data_x_file_name, data_y_file_name)

    # train is only enabled via command line
    unet.train( model_params["batch_size"],
                model_params["epochs"],
                hyper_params["learning_rate"],
                model_path)


if args.segment:

    unet = load_unet(data_x_file_name, data_y_file_name)

    # step 3 - segment images
    segment_images( unet,
                    data_params["labelled_validate_db"],
                    data_params["labelled_validate_dir"],
                    data_params["segmented_dir"],
                    hyper_params["sliding_window_step"],
                    model_path,
                    model_params["batch_size"],
                    hyper_params["max_testing_images_count"],
                    hyper_params["padding_to_remove"])

if args.count:
    # step 4 - count bees on segmented images
    count_bees_on_segmented_images( data_params["labelled_validate_db"],
                                    data_params["labelled_validate_dir"],
                                    data_params["segmented_dir"],
                                    hyper_params["sliding_window_step"],
                                    hyper_params["max_testing_images_count"])

print("Finished")
