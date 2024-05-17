from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load the dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# Normalize pixel data to be in [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Convert integers to a 10-row binary vector
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Create the CNN model
model = Sequential(
    []
)
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
