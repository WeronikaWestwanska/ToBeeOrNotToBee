import tensorflow
import sqlite3
import numpy
import PIL
import os
from keras.preprocessing.image import ImageDataGenerator
from random import randint
import keras
import glob

from FileTools import empty_or_create_directory

from BeesHeatMap import RadialBee
from BeesHeatMap import BeesHeatMap
from BeesDataReader import BeesDataReader

class BeesDataGenerator(BeesDataReader):

    #---------------------------------------------------------
    # writes training x data (color images as numpy)
    # args:
    # data_x_file_name - location where to data X
    # height - height of a typical image with bees
    # width - width of a typical image with bees
    # max_training_images_count - maximum training images, 
    #                    if -1 then take them all
    #---------------------------------------------------------
    def write_training_x_data(self, data_x_file_name, height, width, max_training_images_count):

        # setup numpy array to store bees heatmaps
        images_total_count = len(self.images_dict.keys())

         # check if we are not exceeding maximum images count
        if max_training_images_count != -1 and max_training_images_count < images_total_count:
            images_total_count = max_training_images_count

        x_data = numpy.zeros(shape = (images_total_count, height, width, 3))

        # setup empty directory
        empty_or_create_directory(data_x_file_name)

        # go through collection of images and 
        # store them via augmentation
        processed_images_count = 0
        for image_name, bees_positions_dict in self.images_dict.items():

            image = PIL.Image.open(image_name)
            # normalising the data for 3 channels
            image_as_array = numpy.asarray(image) / 255.0
            numpy.copyto(x_data[processed_images_count], image_as_array)     

            processed_images_count += 1

            # check if we are not exceeding maximum images count
            if processed_images_count >= images_total_count:
                break
                    
        numpy.save(data_x_file_name, x_data)

    #-------------------------------------------------------
    # writes training y data (bees heatmaps)
    # images to directory with
    # data_y_file_name - location where to store heat map
    # height - height of a typical image with bees
    # width - width of a typical image with bees
    # bee_radius - radius of a circle around bee
    # min_bee_prob - minimum bee probability
    # max_bee_prob - maximum bee probability
    # max_training_images_count - maximum training images, 
    #                    if -1 then take them all
    #-------------------------------------------------------
    def write_training_y_data(self,
                              data_y_file_name,
                              height, width,
                              bee_radius,
                              min_bee_prob,
                              max_bee_prob,
                              max_training_images_count):

        # setup numpy array to store bees heatmaps
        images_total_count = len(self.images_dict.keys())

        # check if we are not exceeding maximum images count
        if max_training_images_count != -1 and max_training_images_count < images_total_count:
            images_total_count = max_training_images_count

        y_data = numpy.zeros(shape = (images_total_count, height, width, 2))

        # setup empty directory
        empty_or_create_directory(data_y_file_name)

        # go through collection of images and 
        # store them via augmentation
        processed_images_count = 0
        radial_bee = RadialBee(min_bee_prob, max_bee_prob, bee_radius)
                              
        for image_name, bees_positions_list in self.images_dict.items():

            current_bees_heatmap = BeesHeatMap(radial_bee, bees_positions_list, height, width)

            current_bees_heatmap_array = current_bees_heatmap.get_heatmap()

            # copy contents of the current_bees_heatmap_array to training_y_values
            y_data[processed_images_count, :, :, 0] = 1.0 - current_bees_heatmap_array[:, :]
            y_data[processed_images_count, :, :, 1] = current_bees_heatmap_array[:, :]

            # print("Processed images count = {}".format(processed_images_count))
            processed_images_count += 1

            # check if we are not exceeding maximum images count
            if processed_images_count >= images_total_count:
                break
                    
        numpy.save(data_y_file_name, y_data)
