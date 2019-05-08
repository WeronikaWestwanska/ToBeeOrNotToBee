import matplotlib.pyplot as plt
import scipy.misc
from random import randint
from PIL import Image
import colorsys
import numpy
import glob
import os

from Parameters import data_params
from Parameters import model_params
from Parameters import hyper_params
from BeesDataGenerator import BeesDataGenerator
from FileTools import empty_or_create_directory
from BeesDataTester import BeesDataTester
from Unet import Unet
from FileTools import empty_or_create_directory

#------------------------------------------------------
# generate training data
# args:
# labelled_train_db - path to train db with labels
# labelled_train_dir - path to directory with labelled 
#                      train images
# data_x_file_name - data X file name
# data_y_file_name - data Y file name
# height - height of a training image
# width - width of a training image
# bee_radius - radius of circle simulating bee heatmap
# min_bee_prob - minimum bee probability
# max_bee_prob - maximum bee probability
# max_training_images_count - maximum training images,
#                             if -1 then take them all
#------------------------------------------------------
def generate_data(labelled_train_db,
                  labelled_train_dir,
                  data_x_file_name,
                  data_y_file_name,
                  height,
                  width,
                  bee_radius,
                  min_bee_prob,
                  max_bee_prob,
                  max_training_images_count):

    bees = BeesDataGenerator(labelled_train_db, labelled_train_dir)

    bees.read_db()

    bees.write_training_x_data(data_x_file_name, height, width, max_training_images_count)
    bees.write_training_y_data(data_y_file_name, height, width, bee_radius,
                               min_bee_prob, max_bee_prob, max_training_images_count)

#------------------------------------------------------
# returns full path to segmented and grey image
# segmented_dir - directory with segmentation 
#                      results
# image_file_name - original path
# step - segmentation step
#------------------------------------------------------
def get_segmented_and_grey_image_file_name(segmented_dir, image_file_name, step):

    short_image_file_name = os.path.basename(image_file_name)
    short_image_grey_file_name = short_image_file_name.replace(".jpg", "_segmented_grey_{}.png".format(step))
    short_image_binary_file_name = short_image_file_name.replace(".jpg", "_segmented_binary_{}.png".format(step))
    segmented_grey_file_name = "{}{}".format(segmented_dir, short_image_grey_file_name)
    segmented_binary_file_name = "{}{}".format(segmented_dir, short_image_binary_file_name)
    segmented_partially_name_prefix = "{}{}".format(segmented_dir, short_image_file_name.replace(".jpg", ""))

    return (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix)

#--------------------------------------------------------
# segment images
# unet - implementation of unet
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - images to segment
# segmented_dir - output dir for segmented images
# step - step size for offsetting windows
# model_path - path from which to load model weights
# batch_size - batch size
# max_labelled_images_count - maximum labelled images,
#                             if -1 then take them all
# padding_to_remove - how many pixels to avoid on border
#--------------------------------------------------------
def segment_images(unet,
                   labelled_validate_db,
                   labelled_validate_dir,
                   segmented_dir,
                   step,
                   model_path,
                   batch_size,
                   max_labelled_images_count,
                   padding_to_remove):

    unet.model.load_weights(model_path)
    bees = BeesDataTester(labelled_validate_db, labelled_validate_dir)
    empty_or_create_directory(segmented_dir)
    images_count = 0

    for image_file_name, bees_positions_list in bees.images_dict.items():

        (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix) = \
            get_segmented_and_grey_image_file_name(segmented_dir, image_file_name, step)

        print('input_file_name = {}, output_grey_file_name = {}, output_binary_file_name = {}'.
              format(image_file_name, segmented_grey_file_name, segmented_binary_file_name))

        unet.segment(image_file_name,
                     segmented_grey_file_name,
                     segmented_binary_file_name,
                     segmented_partially_name_prefix,
                     batch_size,
                     padding_to_remove,
                     step)

        images_count += 1
        if max_labelled_images_count != -1 and images_count >= max_labelled_images_count:
            # check if we are not exceeding maximum images count
            break

#------------------------------------------------------
# counts bees on segmented images
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - path to directory with labelled 
#                         validation images
# segmented_dir - output dir for segmented images
# step - step used
# max_labelled_images_count - maximum testing images,
#                             if -1 then take them all
#------------------------------------------------------
def count_bees_on_segmented_images(labelled_validate_db,
                                   labelled_validate_dir,
                                   segmented_dir,
                                   step,
                                   max_labelled_images_count):

    bees = BeesDataTester(labelled_validate_db, labelled_validate_dir)

    # list used for appropriately scaling each image's
    # count into total accuracy
    scaling_list = list()

    # count how many bees in total in test db
    total_bees = 0
    images_count = 0

    for image_file_name, bees_positions_list in bees.images_dict.items():
        total_bees += numpy.float32(len(bees_positions_list))

        images_count += 1
        if max_labelled_images_count != -1 and images_count >= max_labelled_images_count:
            # check if we are not exceeding maximum images count
            break

    total_error = numpy.float32(0.0)
    images_count = 0

    for image_file_name, bees_positions_list in bees.images_dict.items():

        count_from_db = len(bees_positions_list)

        (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix) = \
            get_segmented_and_grey_image_file_name(segmented_dir, image_file_name, step)
        count_from_segment = bees.count_blobs_from_image(segmented_binary_file_name)

        # calculate current image contribution to total_accuracy
        current_error = numpy.abs(numpy.float32(count_from_segment) - numpy.float32(count_from_db)) / \
            numpy.float32(count_from_db)
        total_error += numpy.abs(numpy.float32(count_from_segment) - numpy.float32(count_from_db))
  
        print("current image name = {}, count manual = {}, count automatic = {}, current error = {}".
              format(image_file_name, count_from_db, count_from_segment, current_error))

        images_count += 1
        if max_labelled_images_count != -1 and images_count >= max_labelled_images_count:
            # check if we are not exceeding maximum images count
            break

    total_error /= total_bees
    print("total_error is: {} %".format(total_error * numpy.float32(100)))

#------------------------------------------------------
# save bees color and B&W circled bees images
# images_count - how many random images pairs
# data_x - 4D tensor with training images
# data_y - 4D tensor with B&W destination images
# generated_images_dir - directory where to save images
#------------------------------------------------------
def save_images(images_count, data_x, data_y, generated_images_dir):

    # a sample file name just to create directory if it does not exist
    empty_or_create_directory('{}/file.jpg'.format(generated_images_dir))

    for i in range(0, images_count):
        index = randint(0, data_x.shape[0] - 1)
        print("Index is {}".format(index))

        # outfile_x is an original colour random file
        scipy.misc.imsave('{}/outfile_{}_x.jpg'.format(generated_images_dir, index), data_x[index] * 255.0)
        image_y = Image.fromarray(numpy.uint8(data_y[index, :, :, 1] * 255.0), mode = 'L')
        # outfile_y is a B&W file with circles centred where bees are for the random file above
        scipy.misc.imsave('{}/outfile_{}_y.jpg'.format(generated_images_dir, index), image_y)

#------------------------------------------------------
# save bees/non bees images
# images_count - how many random images pairs
# data_x - 4D tensor with training images
# data_y - 4D tensor with B&W destination images
#------------------------------------------------------
def save_images_overlay(images_count, data_x, data_y):

    for i in range(0, images_count):
        index = randint(0, data_x.shape[0])
        print("Index is {}".format(index))

        data_x[ : , :, :, 1] = data_y[ : , : , : , 1]
        scipy.misc.imsave('outfile_x_{}.jpg'.format(index), data_x[index] * 255.0)

#------------------------------------------------------
# prints parameters used in training/segmentation
# data_params - data parmeters
# model_params - model parameters
# hyper_params - hyper parameteres
#------------------------------------------------------
def print_parameters(data_params, model_params, hyper_params):

    print("Data Parameters:")
    for key, value in data_params.items():
        print("Key: {}, Value: {}".format(key, value))

    print("Model Parameters:")
    for key, value in model_params.items():
        print("Key: {}, Value: {}".format(key, value))

    print("Hyper Parameters:")
    for key, value in hyper_params.items():
        print("Key: {}, Value: {}".format(key, value))

#------------------------------------------------------
# loads unet
# data_x_file_name - 4D tensor with training images
# data_y_file_name - 4D tensor with B&W destination 
#                    generated images
#------------------------------------------------------
def load_unet(data_x_file_name, data_y_file_name):
    unet = Unet(data_x_file_name,
                data_y_file_name,
                data_params["height"],
                data_params["width"],
                hyper_params["window_size"],
                hyper_params["windows_per_image_on_average"],
                hyper_params["percentage_train"],
                hyper_params["min_bee_prct_window"],
                hyper_params["bee_window_percentage"],
                hyper_params["dropout"],
                hyper_params["filters_count"],
                hyper_params["kernel_size"])

    return unet
