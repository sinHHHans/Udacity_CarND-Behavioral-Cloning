'''
This file handles creating a model. This includes training it and storing it as a file.

'''


from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Conv2D, Dropout, Activation, Dense, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split

import csv
import cv2
import numpy as np
import sklearn.utils
import os
import shutil
import random
import matplotlib.pyplot as plt

# *************** RAW ********************************
T1_REGULAR = ".//recorded_data//track_1//regular_driving//driving_log.csv"
T1_RECOVERY = ".//recorded_data//track_1//recovery_driving//driving_log.csv"
T1_SINE = ".//recorded_data//track_1//sine_driving//driving_log.csv"
T1_REVERSE = ".//recorded_data//track_1//reverse_driving//driving_log.csv"
T1_REGULAR_SMOOTH = ".//recorded_data//track_1//regular_driving_smooth//driving_log.csv"


T2_REGULAR = ".//recorded_data//track_2//regular_driving//driving_log.csv"
T2_RECOVERY = ".//recorded_data//track_2//recovery_driving//driving_log.csv"
T2_SINE = ".//recorded_data//track_2//sine_driving//driving_log.csv"
T2_REVERSE = ".//recorded_data//track_2//reverse_driving//driving_log.csv"
# *****************************************************
csv_files_for_training = [T1_REGULAR, T1_RECOVERY, T1_REVERSE, T1_REGULAR_SMOOTH]
batch_size = 32


def prefilter(csv_path_list, plot=True):
    """
    This file filters the paths from the csvs. The idea is that the images that represent straight driving are too many.
    This is why they will be discarded randomly, that in the end there is a reasonable number of straight images.
    The function creates a file that only includes these files that shall be used for training.

    They end with _filtered.csv

    :param csv_path_list: A list that includes paths to csv files. Within these files the filtering will be done.
    :param plot: If true, the results will be plotted. Defaults to True.

    :return: Nothing
    """
    # The straight images are too many. Some filtering needs to be done. I do it by removing SOME files from the csv
    # First, see how big the indifference is
    angles = []
    num_bins = 49
    for path in csv_path_list:
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                angle = float(line[3])
                angles.append(angle)
    bins = np.histogram(angles, bins=num_bins)
    max_index = np.argmax(bins[0])
    bins_sorted = np.copy(bins[0])
    bins_sorted.sort()
    num_going_straight = bins_sorted[-1]
    next_biggest_bin = bins_sorted[-2]
    ratio = num_going_straight / next_biggest_bin
    zero_border = bins[1][max_index:max_index + 2]

    if plot:
        plt.hist(angles, bins=num_bins, fc=(1, 0, 0, 0.3))

    filtered_angles = []
    for path in csv_path_list:
        # Second, remove the straight lines that they are as much as the second biggest group
        out_csv = os.path.normpath(path).replace(path.split('//')[-1],
                                                 path.split('//')[-1].split('.')[0] + '_filtered.csv')
        with open(path, 'r') as csvfile, open(out_csv, 'w') as outputfile:
            reader = csv.reader(csvfile)
            writer = csv.writer(outputfile, lineterminator='\n')
            for row in reader:
                # If the turning angle is around 0, only use it with
                if zero_border[0] <= float(row[3]) <= zero_border[1]:
                    if random.random() <= 1. / ratio:
                        filtered_angles.append(float(row[3]))
                        writer.writerow(row)
                else:
                    writer.writerow(row)
                    filtered_angles.append(float(row[3]))

    if plot:
        plt.hist(filtered_angles, bins=num_bins, fc=(0, 0, 1, 0.3))
        filtered_angles = np.asarray(filtered_angles)
        filtered_angles_flipped = np.concatenate((filtered_angles, filtered_angles * -1))
        plt.hist(filtered_angles_flipped, bins=num_bins, fc=(0, 1, 0, 0.3), edgecolor='black')
        plt.show()


def preprocess_image(img):
    """
    This file takes in an image, increases the contrast, then blurs it and returns the result.

    Took the contrast function implementation more or less from here,
    https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c"

    :param img: Input image to be processed
    :return: processed image
    """
    kernel = np.ones((3, 3), np.float32) / 9
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=6., tileGridSize=(25, 25))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    contrast_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    # cv2.imshow("Contrast", contrast_image)
    # cv2.waitKey(0)
    center_image_blur = cv2.filter2D(contrast_image, -1, kernel)
    # cv2.imshow("Blurred", center_image_blur)
    # cv2.waitKey(0)
    return center_image_blur


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        # Only take half the batch size, as the images will be doubled due to flipping
        for offset in range(0, num_samples, batch_size // 2):

            batch_samples = samples[offset:offset + batch_size // 2]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)

                center_image_blur = preprocess_image(center_image)

                center_angle = float(batch_sample[3])
                images.append(center_image_blur)
                angles.append(center_angle)
                # Add flipped images too
                images.append(np.fliplr(center_image_blur))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model():
    """
    This creates a keras Sequential model and returns it. For details see the write_up.md file.
    :return: The compiled model
    """
    # set up cropping2D layer
    model = Sequential()

    # Remove areas of image that are not useful
    model.add(Cropping2D(cropping=((65, 40), (0, 0)), input_shape=(160, 320, 3)))

    ch, row, col = 3, 55, 320  # Trimmed image format

    # Normalize the images
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 255. - 0.5,
                     input_shape=(col, row, ch),
                     output_shape=(col, row, ch)))

    # Specify the actual network
    model.add(Conv2D(filters=8, kernel_size=(11, 11), strides=(1, 1), padding='valid', activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    model.add(Flatten())
    # model.add(Dropout(0.5))

    model.add(Dense(150))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', )

    return model


def train_model(model, csv_path_list, plot=True):
    """
    This trains the model using the recorded data and returns the trained model.

    :param model: Input model. Expects a sequential keras model that is already compiled.
    :param csv_path_list: A list to csv files that include the paths to all images and their respective steering angles.
    :param plot: If true, plots the results. Defaults to true
    :return: A trained model
    """
    # First extract all the image paths that shall be used to train the model
    lines = []
    for path in csv_path_list:
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

    # Next, create samples
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    history_object = model.fit_generator(train_generator, steps_per_epoch=2 * len(train_samples) // batch_size,
                                         validation_data=validation_generator,
                                         validation_steps=2 * len(validation_samples) // batch_size, epochs=4,
                                         verbose=1)

    if plot:
        # plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    return model


if __name__ == '__main__':
    # Create the model
    prefilter(csv_files_for_training)

    model = create_model()

    # Use filtered data:
    filtered_csvs = [
        os.path.normpath(path).replace(path.split('//')[-1], path.split('//')[-1].split('.')[0] + '_filtered.csv') for
        path in csv_files_for_training]

    # , to train the model:
    trained_model = train_model(model, filtered_csvs)

    # Then save it for the simulator
    trained_model.save('model.h5')
