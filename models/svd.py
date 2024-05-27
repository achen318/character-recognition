import pickle

import matplotlib.pyplot as plt
import numpy as np

from models.base_model import BaseModel


class SVD(BaseModel):
    def __init__(self, k):
        super().__init__(f"svd_{k}.model")
        self.k = k

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)

        except FileNotFoundError:
            mean_matrix = np.mean(trainX, axis=0)
            X = trainX - mean_matrix

            U, S, V = np.linalg.svd(X)

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def display(self) -> None:
        # Show the coefficients/weights matrix
        plt.imshow(self.model.reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.show()

    def predict(self, mat) -> str:
        X = mat.reshape(1, -1)
        Y = X @ self.model  # returns a decimal

        return round(Y[0])
