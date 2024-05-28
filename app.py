import base64

import cv2
from emnist import extract_training_samples, extract_test_samples
from flask import Flask, render_template, request
import numpy as np

from models.bogus import Bogus  # ~ 2.13% accurate
from models.least_squares import LeastSquares  # 2.41% accurate
from models.mean_matrix import MeanMatrix  # 27.86% accurate
from models.mean_value import MeanValue  # 2.13% accurate
from models.neural_network import NeuralNetwork  # ~ 84.64% accurate

# Initialize the Flask app
app = Flask(__name__)

# Initialize the models
bg = Bogus()
ls = LeastSquares()
mm = MeanMatrix()
mv = MeanValue()
nn = NeuralNetwork()

# Load the dataset
trainX, trainY = extract_training_samples("balanced")
testX, testY = extract_test_samples("balanced")

# Normalize pixel data to be float32 in [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Train the models
bg.train(trainX, trainY)
ls.train(trainX, trainY)
mm.train(trainX, trainY)
mv.train(trainX, trainY)
nn.train(trainX, trainY)

# Test the models
bg_acc = f"{(100 * bg.test(testX, testY)):.2f}"
ls_acc = f"{(100 * ls.test(testX, testY)):.2f}"
mm_acc = f"{(100 * mm.test(testX, testY)):.2f}"
mv_acc = f"{(100 * mv.test(testX, testY)):.2f}"
nn_acc = f"{(100 * nn.test(testX, testY)):.2f}"


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
class_labels = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "a",
    37: "b",
    38: "d",
    39: "e",
    40: "f",
    41: "g",
    42: "h",
    43: "n",
    44: "q",
    45: "r",
    46: "t",
}


@app.post("/predict")
def predict():
    img = base64.b64decode(request.data)  # decode the base64 image
    img = np.frombuffer(img, dtype=np.uint8)  # convert to numpy array
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)  # convert to grayscale
    img = cv2.resize(img, (28, 28))  # resize to 28x28
    img = np.invert(img)  # invert the image
    img = img.astype("float32") / 255.0  # normalize the image

    bg_pred = class_labels[bg.predict(img)]
    ls_pred = class_labels[ls.predict(img)]
    mm_pred = class_labels[mm.predict(img)]
    mv_pred = class_labels[mv.predict(img)]
    nn_pred = class_labels[nn.predict(img)]

    # Return the predictions
    return {"bg": bg_pred, "ls": ls_pred, "mm": mm_pred, "mv": mv_pred, "nn": nn_pred}


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
