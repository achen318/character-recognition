import base64

import cv2
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.datasets import mnist

from models.least_squares import LeastSquares  # 24.33% accurate
from models.mean_matrix import MeanMatrix  # 69.68% accurate
from models.mean_value import MeanValue  # 9.8% accurate

# Initialize the Flask app
app = Flask(__name__)

# Initialize the models
ls = LeastSquares()
mm = MeanMatrix()
mv = MeanValue()

# Load the dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# Normalize pixel data to be float32 in [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Convert labels to strings
trainY_str = trainY.astype(str)
testY_str = testY.astype(str)

# Train the models
ls.train(trainX, trainY)
mm.train(trainX, trainY_str)
mv.train(trainX, trainY_str)

# Test the models
ls_acc = 100 * ls.test(testX, testY)
mm_acc = 100 * mm.test(testX, testY_str)
mv_acc = 100 * mv.test(testX, testY_str)


# Render webpage
@app.route("/")
def index():
    return render_template("index.html", ls_acc=ls_acc, mm_acc=mm_acc, mv_acc=mv_acc)


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
    return {"ls": ls.predict(img), "mm": mm.predict(img), "mv": mv.predict(img)}


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
