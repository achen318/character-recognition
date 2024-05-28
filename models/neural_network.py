import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model

from models.base_model import BaseModel


class NeuralNetwork(BaseModel):
    def __init__(self):
        super().__init__("neural_network.keras")

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            self.model = load_model(self.model_file)

        except OSError:
            # Create the model
            self.model = Sequential(
                [
                    Conv2D(32, 3, input_shape=(28, 28, 1)),
                    MaxPooling2D(2, 2),
                    Flatten(input_shape=(28, 28, 1)),
                    Dense(128, activation="relu"),
                    Dense(47, activation="softmax"),
                ]
            )

            # Compile and train the model
            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self.model.fit(trainX, trainY, epochs=5)

            # Save the model
            self.model.save(self.model_file)

    def predict(self, mat) -> int:
        pred = self.model.predict(mat.reshape(1, 28, 28))
        return int(np.argmax(pred))

    def test(self, testX, testY) -> float:
        return self.model.evaluate(testX, testY)[1]
