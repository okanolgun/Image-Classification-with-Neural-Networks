import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()   
# The tuples we mentioned as training_images and testing_images
# are already in the keras package. We get them with the load_data method.

training_images, testing_images = training_images / 255, testing_images / 255
# normally the pixels on the training and test data are from zero to 255.
# Here we reduced the pixels to between zero and one and did this
# so that the model gives better, faster and more beautiful results.

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()
# This code visualizes the 16 images from the training dataset
# in black and white on a 4x4 grid, removes the axis markers,
# and adds the class label below each image so that images and labels can be quickly inspected

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


model = models.load_model('image_classifier_model')

img = cv.imread('image_classifier_model/image.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)
# the visualising the image

prediction = model.predict(np.array([img]) / 255)
# doing the prediction.
# we need to pass numpy array. because our model has certain structure

index = np.argmax(prediction)
# we need to have max value with index
# because one of the neuron will have max activation and we want it
print(f'prediction is : {class_names[index]}')


