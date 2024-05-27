from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from models.least_squares import LeastSquares  # 24.33% accurate
from models.mean_matrix import MeanMatrix  # 69.68% accurate
from models.mean_value import MeanValue  # 9.8% accurate
from models.neural_network import NeuralNetwork
from models.svd import SVD

# Initialize the model
model = NeuralNetwork([784, 128, 10], 1000, 0.05)

# Load the dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# Normalize pixel data to be float32 in [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# # Convert labels to strings
# trainY = trainY.astype(str)
# testY = testY.astype(str)

trainX = trainX.reshape(trainX.shape[0], -1)
testX = testX.reshape(testX.shape[0], -1)

trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Train the model
model.train(trainX, trainY)

# Test the model
acc = model.test(testX, testY)

# Display results and the model
print(f"{100 * acc}% accurate")
model.display()
