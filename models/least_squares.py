import pickle

import numpy as np

from models.base_model import BaseModel


class LeastSquares(BaseModel):
    def __init__(self):
        super().__init__("least_squares.model")

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)

        except FileNotFoundError:
            # Perform linear regression with pseudo-inverse
            X = trainX.reshape(trainX.shape[0], -1)
            Y = trainY

            B = np.linalg.pinv(X.T @ X) @ X.T @ Y
            self.model = B

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def predict(self, mat) -> int:
        X = mat.reshape(1, -1)
        Y = X @ self.model  # returns a decimal
        Y = np.clip(Y, 0, 46)  # clip to [0, 46]
        return round(Y[0])
