import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
#The tuples we mentioned as training_images and testing_images
# are already in the keras package. We get them with the load_data method.

training_images, testing_images = training_images/255, testing_images/255
#Normally the pixels on the training and test data are from zero to 255.
# Here we reduced the pixels to between zero and one and did this
# so that the model gives better, faster and more beautiful results.

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(training_images[i], cmap= plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()
#This code visualizes the 16 images from the training dataset
# in black and white on a 4x4 grid, removes the axis markers,
# and adds the class label below each image so that images and labels can be quickly inspected.

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]
#Testing a neural network is a difficult and long process for computers.
# To shorten this process, we limited our test images and
# training images from zero to 20,000 and from zero to 4,000.

model = models.Sequential()
#we created a sequential model

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
#32 filters with 3x3 size. ReLU equals the negative values to the 0.
# because we want to help the model learn non-linear relationships

model.add(layers.MaxPooling2D((2,2)))
#this layer does the wownsampling.
# it takes the max values from each 2x2 section. and shrinks feature maps
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))

model.add(layers.Flatten())
#This layer converts the entire 2D matrix into a one-dimensional vector.
# The 2D feature maps obtained from the convolution layers are flattened
# and made ready for the fully connected (dense) layer.

model.add(layers.Dense(64, activation='relu'))
#Dense (Fully Connected) Layer: This layer is a structure in which
# each neuron is fully connected to every neuron in the previous layer.
model.add(layers.Dense(10, activation='softmax'))
#Dense Layer (Output Layer): This layer has 10 neurons. This typically represents a
# 10-class classification problem (e.g., 10 different categories in the CIFAR-10 dataset).
#activation='softmax': This is a commonly used activation function for classification problems.
# softmax gives the probability scores of each class, which add up to 1. It estimates how likely each class is.

