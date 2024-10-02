# Image-Classification-with-Neural-Networks    

In this project, I developed an image classification model using the CIFAR-10 dataset. I created a neural network model using TensorFlow and Keras libraries. Images and labels were taken from the dataset and pixels were normalized between 0 and 1. I designed the model using convolutional neural network (CNN) architectures such as Conv2D (convolutional layers) and MaxPooling2D (maximum pooling layers) to classify 32x32 color images. ReLU activation function is used to cancel out negative values ​​and accelerate learning, while Softmax function is used to perform the classification process in the output layer. The model was trained with the adam optimization algorithm and sparse_categorical_crossentropy loss function, and a classification process was performed into 10 classes (such as airplane, ship, truck). The performance of the model was measured with the accuracy rate on the test data and the results were recorded. The model was recorded at the end of the training process and stored for future use.
