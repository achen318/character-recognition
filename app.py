import base64

import cv2
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.datasets import mnist

from models.least_squares import LeastSquares  # inconclusive
from models.mean_matrix import MeanMatrix  # 69.68% accurate
from models.mean_value import MeanValue  # 9.8% accurate

# Initialize the Flask app
app = Flask(__name__)

# Initialize the models
mm = MeanMatrix()
mv = MeanValue()

# Load the dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# Normalize pixel data to be float32 in [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Convert labels to strings
trainY = trainY.astype(str)
testY = testY.astype(str)

# Train the models
mm.train(trainX, trainY)
mv.train(trainX, trainY)

# Test the models
mm_acc = mm.test(testX, testY)
mv_acc = mv.test(testX, testY)


# Render webpage
@app.route("/")
def index():
    return render_template("index.html", mm_acc=mm_acc, mv_acc=mv_acc)


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
    return {"mm": mm.predict(img), "mv": mv.predict(img)}


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
