# Title:
To Bee Or Not To Bee 

# Short description

This code is a supplement to the article by Weronika W. Westwańska and Jerzy S. Respondek titled:
"Counting Instances of Object in Colour Images Using U-Net Network on Example of Honey Bees"

The goal of this work was to examine a novel method for counting instances of Object Of Interest (OOI) in colour digital images, using modern Deep Learning technique called U-Net network. The method works on approximate shaping of OOIs into circles, further modelling data generation, segmentation and counting the OOIs. We also propose a new method for segmentation of OOI, based on a trained U-Net network, which is applied to produce a set of images converted to an OOI heatmap. The heatmap is later binarised for a purpose of OOI counting.

The data comes from https://www.kaggle.com/jonathanbyrne/to-bee-or-not-to-bee. We used original set of 550 labelled images, further combined with 1086 images labelled by us, to create conditions for conducting experiments on how our approach in U-Net modelling would influence bees counting error.

Details of the working of this method are clearly described in the source code comments.

# Description of project contents:

- `data/labelled.train` - images used in training U-Net network,
- `data/labelled.validate` - images used in calculation of relative bees counting error
- `data/labels.train.db` - SQLite3 database with coordinates of bees images from data\labelled.train directory
- `data/labels.validate.db` - SQLite3 database with coordinates of bees images from data\labelled.validate directory
- `results/set1`- 25 different experiments for parameters set 1 (mentioned below) attempting to find how modelling set size influences OOI counting error. For each of 5 subsets for sizes (100, 200, 500, 1000 and 1096) only the best training model was picked up and used for segmentation.
- `results/set2` - 25 different experiments for parameters set 2 (mentioned below) attempting to find how modelling set size influences OOI counting error. For each of 5 subsets for sizes (100, 200, 500, 1000 and 1096) only the best training model was picked up and used for segmentation.
- `.editorconfig` - settings file for Visual Studio.
- `BeesDataGenerator.py` - routines used in modelling data generation,
- `BeesDataReader.py` - routines for reading SQLite3 database and its contents where bees coordinates are stored
- `BeesDataTester.py` - routines for calculating OOIs,
- `BeesHeatMap.py` - routines for OOI heatmap generation,
- `experiment.sh` - simple bash script to automate process of training,
- `FileTools.py` - routines for creating directory, 
- `Parameters.py` - parameters values used in training and segmentation,
- `readme.md` - this file,
- `ToBeeOrNotToBee.sln` - Visual Studio python project file,
- `ToBeeOrNotToBee.pyproj` - Visual Studio python project file,
- `ToBeeOrNotToBee.pyperf` - Visual Studio python project file,
- `Tools.py` - various routines for training, segmentation, saving data
- `Unet.py` - routines for U-Net modelling,

Note: Please bear in mind that what we call in the Python code as train set is an equivalent to modelling set in the article, and validate set is an equivalent to segmentation set in the article. This confusion is caused by naming conventions made at a time of starting the work, which was later clarified when summarising it in a form of the article.

# Results

The most time consuming part of the process is OOI error calculation. The training of U-Net takes about 5 minutes of a PC with NVidia 1080 GTX with 8GB of memory. The segmentation took about 2h 40 minutes to process 540 images in the segmentation set.
We decided to train the U-Net network for 2 different sets of parameters describing size of a circle approximating a generic bee's body.

**Set 1** - circle of bee radius = 20, with Pmin = 0.80, Pmax = 1.00, amount of random training windows per image = 60, minimum percentage of pixels per window to be considered as a bee = 45
**Set 2** - circle of bee radius = 16, with Pmin = 0.99, Pmax = 1.00, amount of random training windows per image = 80, minimum percentage of pixels per window to be considered as a bee = 50

For each of the sets parameters (set1, set 2) we run the script (on a Cygwin Windows 10 machine):
```sh
./experiments.sh
```
This script generated a set of 25 directories with different suffixes, where a log file and U-Net trained model was stored. The 25 directories would represent training result of U-Net where the modelling sets of size 100, 200, 500, 1000 and 1096 was used to train U-Net network 5 times each. 
Once we collected the log files, we would pick the best trained models, and use them for segmenting and further calculation of relative bees counting error.
The best model was found for Set 1, modelling set equal to 1096 images, where the error = 14.28 %.

Below is a table representing training validation accuracy and OOI relative error in percentage points performed for Parameters Set 1.

| Set 1 | Modelling set size | Training Validation Accuracy | OOI Error |
| ----- | ------------------ | ---------------------------- | --------- |
| #1 | 100 | 95.13 | 75.08 % |
| #2 | 200 | 94.65 | 73.00 % |
| #3 | 500 | 94.25 | 24.23 % |
| #4 | 1000 | 94.93 | 16.83 % |
| #5 | 1096 | 94.68 |14.27 % |

* Data for the best result of 14.27% is available in `results/set1/SET1_20190428_221502_FILES_COUNT_1096`.
* Segmented images are available in `results/set1/SET1_20190428_221502_FILES_COUNT_1096/segmented`

Below is a table representing training validation accuracy and OOI relative error in percentage points performed for Parameters Set 2.

| Set 2 | Modelling set size | Training Validation Accuracy | OOI Error |
| ----- | ------------------ | ---------------------------- | --------- |
| #1 | 100 | 96.41 | 75.24 % |
| #2 | 200 | 96.15 | 74.55 % |
| #3 | 500 | 95.24 | 26.35 % |
| #4 | 1000 | 96.23 | 16.87 % |
| #5 | 1096 |  96.12 | 18.17 % |

# How to use

In order to perform a whole end-to-end process of data generation, training, segmentation and counting of OOIs the user needs to decide on values of `Parameters.py` (or leave them how they are by default) and then perform the following commands:
```
python Main.py --generate --train
python Main.py --segment --count
```

The first command will generate modelling data and train UNet network. The results will be stored by default in `data/train_r16` (depending on value of 'bee_radius').
The second command will use model stored in `data/train_r16` and create a new directory `data/segmented`. This bit can take at least 2 hours (depending on value of sliding_window_step in `Parameters.py`) as well as graphics card.

**Note**: Please ensure when training the U-Net network that the process is not stuck at the same validation accuracy (sometimes it can happen due to a bug in Keras). In such case, delete the contents of `data/train_r16` directory and start over. Typical validation accuracy should reach at least 94%.

**Last Modified**: 2019/05/06