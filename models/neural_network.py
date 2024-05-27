import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model

from models.base_model import BaseModel


class NeuralNetwork(BaseModel):
    def __init__(self):
        super().__init__("neural_network.model")

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            self.model = load_model(self.model_file)

        except OSError:
            # Create the model
            self.model = Sequential()
            self.model.add(Flatten(input_shape=(28, 28)))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dense(10, activation="softmax"))

            # Compile and train the model
            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self.model.fit(trainX, trainY, epochs=3)

            # Save the model
            self.model.save(self.model_file)

    def predict(self, mat) -> str:
        pred = self.model.predict(mat.reshape(1, 28, 28))
        return str(np.argmax(pred))

    def test(self, testX, testY) -> float:
        return self.model.evaluate(testX, testY)[1]
