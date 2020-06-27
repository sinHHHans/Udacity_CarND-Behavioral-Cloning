from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from model import csv_files_for_training, generator
import numpy as np
from keras.models import load_model
import csv
from PIL import Image


# Thanks to https://github.com/JGuillaumin/DeepLearning-NoBlaBla/blob/master/KerasViz.ipynb for sharing the below
# this function will plot all the feature maps within a fig of size (20,20)
def plot_feature_maps(feature_maps, title):
    """
    This function plots feature maps of a keras model.
    :param feature_maps:
    :param title:
    :return:
    """
    if len(feature_maps.shape) == 3:

        height, width, depth = feature_maps.shape
        nb_plot = int(np.rint(np.sqrt(depth)))
        fig = plt.figure(figsize=(20, 20))
        for i in range(depth):
            plt.subplot(nb_plot, nb_plot, i+1)
            plt.imshow(feature_maps[:,:,i])
            plt.title('feature map {}'.format(i+1) + " " + title)
        plt.show()


def plot_layer(layer_name, img_, model):
    # check if the layer_name is correct

    features_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = features_extractor.predict(img_)[0]
    print("At layer \"{}\" : {} ".format(layer_name, feature_maps.shape))
    plot_feature_maps(feature_maps)


trained_model = load_model('model.h5')
csv_path_list = csv_files_for_training
lines = []
for path in csv_path_list:
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

# compile and train the model using the generator function
train_generator = generator(lines[0:8], batch_size=8)
img = train_generator.__next__()
imgString = "writeup_images//test_image.jpeg"
# Visualize
layers = np.array([layer.name for layer in trained_model.layers])

for layer in layers:
    image = Image.open(imgString)
    image_array = np.asarray(image)
    extractor = Model(inputs=trained_model.input, outputs=trained_model.get_layer(layer).output)

    layer_features = extractor.predict(image_array[None, :, :, :], batch_size=1)
    plot_feature_maps(layer_features[0], title=str(layer))
