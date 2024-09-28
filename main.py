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
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

