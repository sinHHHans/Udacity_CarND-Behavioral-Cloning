# **Behavioral Cloning** 
In this write up I will face the rubric points and explain my approaches how to predict a steering angle from an image like the following.

![][testimage]


## Rubric points / table of contents:
* How to run the software
* Submission files
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## How to run the software
This section shows how to run the software

### Collect data
Use the Udacity Simulator to record data. I suggest to structure the recorded data in dedicated folders to be able to 
control which data will be used.

### Train the model 

Training the model is done by calling model.py. The training is handled by 'train_model()'.

Call:
<!-- -->
    python model.py

See comments in the code for details.

### Test the model
Call:
<!-- -->
    python drive.py model.h5
    windows_sim

In windows_sim, choose autonomous mode.

#### Submission files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md A written report
* visualize.py A set of functions to visualize the network output
* run_v9.mp4 The video of one round
* requirements.txt The list of packages required for running the scripts

## Use the simulator to collect data of good driving behavior  

### Is the creation of the training dataset and training process documented?
I use the simulator to create data using the 'R' key. I structure the data for several reasons.

I decided to structure my data collection in the following matter:
<!-- language: lang-none -->
                                              +------------------+
                                              |                  |
                                              |  Collected Data  |
                                              |                  |
                                              +---------+--------+
                                                        |
                                    +-------+           |           +-------+
                                    |Track 1| <---------+---------> |Track 2|
                                    +---+---+                       +---+- -+
                                        |                               |
    +----------+---------+----------+---+--------------+ +----------+---+-----+----------+------------------+
    |          |         |          |                  | |          |         |          |                  |
    |  Regular | Reverse | Sine     | Regular driving  | |  Regular | Reverse | Sine     | Regular driving  |
    |  Driving | Driving | Driving  | Smooth           | |  Driving | Driving | Driving  | Smooth           |
    |          |         |          |                  | |          |         |          |                  |
    +----------+---------+----------+------------------+ +----------+---------+----------+------------------+

 
I structure it so I can turn on and off different data sets and see how this affects the result. First let me explain what the categories mean.
##### Regular Driving
A few laps of driving the respective track in a normal fashion, means trying to stay in the center.
##### Reverse driving
A few laps of driving the track, but in the opposite direction to help the model generalize.
##### Sine driving
This is driving with oscilation within the track. Just like a singer might oscilate around the note 'A' to actually sing it, because it is much easier to accurately sing the note as a constant. In the same logic I was hoping the car might have an easier time to learn oscilating aroung the center of the track, than actually staying in the center constantly.
##### Regular driving smooth
This category I implemented later during testing. It is very similiar to regular driving, but tries to drive the track as smooth as possible, trying to avoid very string turning at short times.

I choose the training data as follows:
<!-- language: Python -->
    T1_REGULAR = ".//recorded_data//track_1//regular_driving//driving_log.csv"
    # ...
    csv_files_for_training = [T1_REGULAR, T1_RECOVERY, T1_REVERSE, T1_REGULAR_SMOOTH]
    # ...

## Build, a convolution neural network in Keras that predicts steering angles from images
In order to predict the steering angle from images, the network will in some way need to convolve the images and find characteristics that lead to the lane boundaries.

### Is the model architecture documented?

Here is how my model looks like:
<!-- language: lang-none -->
    +---------------------------------------------------------------+
    |                     Increase contrast                         |
    +----------------------------+----------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |                       Input layer                             |
    +----------------------------+----------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |          Cropping layer (160,320) => (55, 320))               |
    +----------------------------+----------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |                     Normalization Layer                       |
    +----------------------------+----------------------------------+
                                 |
    +---------------------------------------------------------------+
    |                     8 x Conv2D (11,11)                        |
    +---------------------------------------------------------------+
                                 |
    +---------------------------------------------------------------+
    |                    16 x Conv2D (5,5)                          |
    +---------------------------------------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |                      16 x Conv2D (3,3)                        |
    +----------------------------+----------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |                         Flatten                               |
    +----------------------------+----------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |                        150 x Dense                            |
    +----------------------------+----------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |                        Dropout 50%                            |
    +----------------------------+----------------------------------+
                                 |                                 
    +----------------------------v----------------------------------+
    |                       50 x Dense                              |
    +----------------------------+----------------------------------+
                                 |
    +----------------------------v----------------------------------+
    |                        Dropout 20%                            |
    +----------------------------+----------------------------------+
                                 |                                 
    +----------------------------v----------------------------------+
    |                    1 x Output Neuron                          |
    +---------------------------------------------------------------+
    
As a first step, even before the network, I increase the contrast in the images, as I hope that this makes it easier for
the network to find the edges that make up the lane boundaries.

#### Convulutional layers output
In the following, the first layers are demonstrated with an example input:

![][testimage]


First do cropping:

![][cropping]

I skip the normalization layer here because it looks just like above.

The first conv layer introduces a large kernel and naturally finds large features.

![][conv1]

More details are found:

![][conv2]

Even more details are found:

![][conv3]


### Has an appropriate model architecture been employed for the task?
I oriented myself at the VGG network architecture by NVIDIA, but simplified it a bit. 
Especially I chose a rather large kernel in the first conv layer, since I want the lane boundaries to be found. 
As theses are rather large features, I want to guide the net in this directions by large filter kernels.

In the consecutive layers I reduce the kernel size, as the NVIDIA solution does. It it a solution designed for 
autonomous driving, so the orientation should be valid.

The end of the network consists, in that same logic, of fully connected layers, as they combine the found features from the
conv layers in a way that a turning angle can be derived. As the only output it the steering angle, the last layer has only one
output neuron.

### Has an attempt been made to reduce overfitting of the model?
In order to avoid overfitting, I add two Dropout layers. Besides this, I reduce the amount of driving straight images and
I help to generalize the model by using augmented data and data the is generated by driving in the opposite direction.

### Have the model parameters been tuned appropriately?
I use an Adam optimizer, which adapts the learning rate to the progress it makes towards an optimal solution.


## Train and validate the model with a training and validation set

For training and validation I use the data generated above, described in image 1.

### Is the training data chosen appropriately?
I filter and augment the data to get better results.

#### Filtering

When looking at a histogram of the angles included in the test images, it becomes clear that the very most images come
with a turning angle of zero. This is natural, as most of the time driving straight is a good solution to stay in the 
center of the road.

The problem is as a consequence, that my model seemed to have learned that a small turning angle close to zero 
is a good choice in about 96% of the time. I solve this problem by filtering most of the data with turning angle 0.
A challenge was to make sure that the straight parts are not underrepresented afterwards. I solve this with a histogram
as shown in the following example.

![][no_filter]

In order to avoid over- AND underfitting, I choose the number of straight images to be the same as the second highest
bin in the histogram. Again this implies the problem, that deleting the first 'N' images with turning angle close to 0
will result in areas that have no straight images on the track, whereas other areas have still too many images. I solve
this by stochastically omitting the images, with the probability of filtering an image is set in a way, that in the end
the number of randomly chosen images is approximately the same as the second highest bin. The result can be seen in the 
following image:

![][filter]

In code it looks like this:
<!-- language: Python -->
    ratio = num_going_straight / next_biggest_bin
    if random.random() <= 1. / ratio:
        filtered_angles.append(float(row[3]))
        writer.writerow(row)
        
#### Augmentation

I augment the data by mirroring the data from left to right and negating the angle. This way I can double the data and
have a symmetrical distribution of steering angles around 0. The final result that is used for training is shown below.

![][filter_augmentation]
 
 
As suggested by David, using a generator might be a good way to deal with memory constraints, as not all the images have
to be loaded in memory.

My generator uses only the filtered images that have a reasonable amount of straight drives and is based on the 
proposed code of David, however I added the extra functions of augmentation as shown above.

## Test that the model successfully drives around track one without leaving the road

### Is the solution design documented?
The quest of finding a running setup was a long one. In the following I have the (fragmented) notes I took during testing.

| Iteration | Idea                                                                                                                                                                                                           | Training data                                                      | Description                                                                                                                                                                                                                                                                                  | Hyperparameters                                  | Result                                                                                                                                                                                                                                        | Conclusion                                                                                                       |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| 1         | Test the pipeline with no real ambitions                                                                                                                                                                       | T1_REGULAR, T1_REVERSE                                             |  Create a net that does anything, just to test the pipeline.  I train it with one round of regular driving and one round of reverse driving.                                                                                                                                                 | Optimizer: Adam Loss: MSE Epochs: 5              | The vehicle gets till the end of the bridge until it gets stuck. Pipeline works, which is nice.                                                                                                                                               | Probably, everything needs improvement                                                                           |
| 2         | Improve the performance just by more data                                                                                                                                                                      | T1_REGULAR, T1_RECOVERY, T1_REVERSE                                | Same as above but additionally with recovery data.                                                                                                                                                                                                                                           | Optimizer: Adam Loss: MSE Epochs: 5              | Vehicle fails before the bride, new data made it worse.                                                                                                                                                                                       | As the recovery data made it worse, my recovery data set might be bad. I will re-record it.                      |
| 3         | Try again with re-recorded data for recovery. (I think I was too sloppy the first try)                                                                                                                         | T1_REGULAR, T1_RECOVERY, T1_REVERSE                                |  Same as above                                                                                                                                                                                                                                                                               | Same as above                                    | Looks smoother first but fails before the bridge again                                                                                                                                                                                        | Initial net was probably not very good. Improving into having two ConvLayers                                     |
| 4         | Improved network architecture                                                                                                                                                                                  | T1_REGULAR, T1_RECOVERY, T1_REVERSE                                | Added an additional ConvLayer for deeper learning                                                                                                                                                                                                                                            | Same as above                                    | Looks smoother first but fails before the bridge again                                                                                                                                                                                        | Maybe more input data                                                                                            |
| 5         | Same architecture, but using Track1_Sine data too                                                                                                                                                              | T1_REGULAR, T1_RECOVERY, T1_REVERSE, T1_SINE                       | More data, hopefully improves it                                                                                                                                                                                                                                                             | Same as above                                    | Its worse. Vehicle tries to constantly go in the same direction. This makes sense because the oscilating driving behavious assigns turning angles of any size to any of the images, this makes it hard to learn anything only by convolution. |  1) Sine driving was a bad idea 2) A sophisticated network architecture might perform better                     |
| 6         | Try NVidia Architecture                                                                                                                                                                                        | T1_REGULAR, T1_RECOVERY, T1_REVERSE                                | The architecture was designed for this use case, so it should do a good job.                                                                                                                                                                                                                 | Same as above                                    | Runtime error: Not enough memory. My Graphics Card only has 6GB of RAM.                                                                                                                                                                       | Simplify the network or go to AWS                                                                                |
| 7         | Simpler architecture that my Graphics card can handle                                                                                                                                                          | Same as above                                                      | Same as above                                                                                                                                                                                                                                                                                | Same as above                                    | Runs but fails again at the same spot                                                                                                                                                                                                         | Need better data or better model or both.                                                                        |
| 8         | New data in Recovery for the tricky spot                                                                                                                                                                       | Same as above but enhanced                                         | The new data shall do the trick                                                                                                                                                                                                                                                              | Optimizer: Adam Loss: MSE Epochs: 3              | Fails again at the same spot                                                                                                                                                                                                                  | Model has some issue                                                                                             |
| 9         |  Fix: Normalization Increase Cropping                                                                                                                                                                          | Enhance training speed and fix a bug in normalization              | Better performance                                                                                                                                                                                                                                                                           | Same as above                                    | Same as above                                                                                                                                                                                                                                 | Same as above                                                                                                    |
| 10 - 19   | Many tries changing the data, and the model[...]                                                                                                                                                               | Many tries changing the data, and the model[...]                   | Many tries changing the data, and the model[...]                                                                                                                                                                                                                                             | Many tries changing the data, and the model[...] | Many tries changing the data, and the model[...]                                                                                                                                                                                              | Many tries changing the data, and the model[...]                                                                 |
|  20       |  The netowrk still performs very bad in curves. Probably because most of the images show turning angles of close to 0. I will filter them a bit, so only a reasonable split of the data shows driving straight | T1_REGULAR, T1_RECOVERY, T1_REVERSE                                | I create a histogram of the turning angles. The highest peak is obviously the zero angles. I randomly omit theses in a way, that in the end I have the same amount of straight images as images with a slight curve. This way I dont overfit on straight images but also dont underfit them. | Optimizer: ADAM Loss: MSE Epochs: 4              | The car seems to be more agile and willing to turn harder, and it does in fact. Still fails before the bridge though.                                                                                                                         | Something must still be wrong, as after the first epoch the validation error is much higher than training error. |
|  21       | Found a bug. The data was used for training 'batch_size' times too often per epoch.                                                                                                                            | T1_REGULAR, T1_RECOVERY, T1_REVERSE                                | The training must be executable faster.                                                                                                                                                                                                                                                      |  Optimizer: ADAM Loss: MSE Epochs: 4             | Training is done within minutes rather than ~1-2 hours.                                                                                                                                                                                       | Enhance the model and get quicker feedback and less overfitting.                                                 |
|  22 - ~30 | More data on smoother driving shall be used. I drive a few more rounds                                                                                                                                         |  T1_REGULAR, T1_RECOVERY, T1_REVERSE, T1_REGULAR_SMOOTH            | The network still fails, but as it now runs much faster, more data will not hurt.                                                                                                                                                                                                            | Same as above, trying different epochs           | Much better, but often fails at only one spot                                                                                                                                                                                                 | Enhance model and retry.                                                                                         |
| ~30       | After several trials, found a solution.                                                                                                                                                                        | T1_REGULAR, T1_RECOVERY, T1_REVERSE, T1_REGULAR_SMOOTH             | A network simpler than that of NVIDIA, but still similiar.                                                                                                                                                                                                                                   |  Optimizer: ADAM Loss: MSE Epochs: 4             | The network does the job and succeeds in driving the course.                                                                                                                                                                                  | Record and consider it done (hopefully)                                                                          |
|           |                                                                                                                                                                                                                |                                                                    |                                                                                                                                                                                                                                                                                              |                                                  |                                                                                                                                                                                                                                               |                                                                                                                  |

In the end I finally had a working solution. The training curve looks like this:

![][training_curve]

The evaluation images are a subset of the training images, and as they were recorded in a drive the evaluation 
images share many similarities with the training images, which is why it performs quite well even after the first epoch.

The video shows my result:

<video width="320" height="160" controls>
  <source src="run_v9.mp4" type="video/mp4">
</video>

If there is not video above, markdown does not include my html. In that case the video can be found in this same
directory and is called run_v9.mp4


[//]: # (Image References)

[no_filter]: ./writeup_images/bins_no_filter.png "No filtering"
[filter]:    ./writeup_images/bins_filter.png "Filtering"
[filter_augmentation]:    ./writeup_images/bins_filter_augmentation.png "Filtering and Augmentation"
[testimage]:    ./writeup_images/test_image.jpeg "Filtering and Augmentation"
[training_curve]:    ./writeup_images/Learning_curve.png "Filtering and Augmentation"

[cropping]:    ./writeup_images/Cropping.png "Cropping"
[conv1]:    ./writeup_images/Conv_Layer1.png "Conv 1"
[conv2]:    ./writeup_images/Conv_Layer2.png "Conv 2"
[conv3]:    ./writeup_images/Conv_Layer3.png "Conv 3"
