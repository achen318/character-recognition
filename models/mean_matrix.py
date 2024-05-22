import numpy as np
import pickle

from models.base_model import BaseModel

class MeanMatrix(BaseModel):
    def __init__(self):
        super().__init__("mean_matrix.model")

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)

        except FileNotFoundError:
            # Train the model
            for mat, char in zip(trainX, trainY):
                if char not in self.model:
                    self.model[char] = np.zeros(mat.shape)

                # Accumulate the mean of matrices
                self.model[char] += mat / len(trainX)

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def predict(self, mat) -> str:
        closest_char = ""
        closest_dist = np.inf

        for char, char_mean in self.model.items():
            # Minimize the Frobenius norm of the difference in matrices
            dist = np.linalg.norm(mat - char_mean)

            if dist < closest_dist:
                closest_dist = dist
                closest_char = char

        return closest_char
