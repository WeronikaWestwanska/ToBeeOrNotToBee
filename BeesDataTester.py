from skimage import io
from scipy import ndimage
from skimage import measure

from BeesDataReader import BeesDataReader

class BeesDataTester(BeesDataReader):

    #------------------------------------------------------
    # ctor
    #------------------------------------------------------
    def __init__(self, db_name, labelled_dir):
        self.db_name = db_name
        self.labelled_dir = labelled_dir
        self.read_db()

    #------------------------------------------------------
    # Counts occurences of white blobs in black and
    # white image, where black is background and white
    # is the object of interest
    # image_path - image path
    #------------------------------------------------------
    def count_blobs_from_image(self, image_path):

        im = io.imread(image_path)
        drops = ndimage.binary_erosion(im)
        labels = measure.label(drops)

        return labels.max()

    #------------------------------------------------------
    # returns dictionary with file name and blobs count
    # segmented_dir - output dir for segmented images
    # step - step used in segmentation
    #------------------------------------------------------
    def count_blobs_from_images(segmented_dir, step):

        result_dict = dict()

        # go through file names stored in database
        for image_name, bees_positions_dict in self.images_dict.items():

            full_segmented_image_path = "{}_segmented_binary_{}.png".format(image_name, step)
            full_segmented_image_path = "{}{}".format(segmented_dir, full_segmented_image_path)

            blobs_count = self.count_blobs_from_image(full_segmented_image_path)
            result_dict[image_name] = blobs_count

        return result_dict
