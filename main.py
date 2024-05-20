from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from models.simple_mean import SimpleMean

# Load the dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# Normalize pixel data to be float32 in [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Train the model
print(len(trainX))

# Classify the data

# Convert integers to a 10-row binary vector
trainY = to_categorical(trainY)
testY = to_categorical(testY)
