import base64

import cv2
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.datasets import mnist

from models.bogus import Bogus  # ~ 10% accurate
from models.least_squares import LeastSquares  # 24.33% accurate
from models.mean_matrix import MeanMatrix  # 69.68% accurate
from models.mean_value import MeanValue  # 9.8% accurate
from models.neural_network import NeuralNetwork  # 97.32% accurate

# Initialize the Flask app
app = Flask(__name__)

# Initialize the models
bg = Bogus()
ls = LeastSquares()
mm = MeanMatrix()
mv = MeanValue()
nn = NeuralNetwork()

# Load the dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# Normalize pixel data to be float32 in [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Convert labels to strings
trainY_str = trainY.astype(str)
testY_str = testY.astype(str)

# Train the models
bg.train(trainX, trainY_str)
ls.train(trainX, trainY)
mm.train(trainX, trainY_str)
mv.train(trainX, trainY_str)
nn.train(trainX, trainY)

# Test the models
bg_acc = round(100 * bg.test(testX, testY_str), 4)
ls_acc = round(100 * ls.test(testX, testY), 4)
mm_acc = round(100 * mm.test(testX, testY_str), 4)
mv_acc = round(100 * mv.test(testX, testY_str), 4)
nn_acc = round(100 * nn.test(testX, testY), 4)


# Render webpage
@app.route("/")
def index():
    return render_template(
        "index.html",
        bg_acc=bg_acc,
        ls_acc=ls_acc,
        mm_acc=mm_acc,
        mv_acc=mv_acc,
        nn_acc=nn_acc,
    )


# Image via POST request -> prediction response
@app.post("/predict")
def predict():
    img = base64.b64decode(request.data)  # decode the base64 image
    img = np.frombuffer(img, dtype=np.uint8)  # convert to numpy array
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)  # convert to grayscale
    img = cv2.resize(img, (28, 28))  # resize to 28x28
    img = np.invert(img)  # invert the image
    img = img.astype("float32") / 255.0  # normalize the image

    # Return the predictions
    return {
        "bg": bg.predict(img),
        "ls": ls.predict(img),
        "mm": mm.predict(img),
        "mv": mv.predict(img),
        "nn": nn.predict(img),
    }


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
