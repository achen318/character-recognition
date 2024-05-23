import numpy as np
import pickle

from models.base_model import BaseModel


class SVD(BaseModel):
    def __init__(self, k: int):
        super().__init__("svd.model")
        self.k = k

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)

        except FileNotFoundError:
            # Train the model
            for mat, label in zip(trainX, trainY):
                if label not in self.model:
                    self.model[label] = np.zeros(mat.shape)

                # Accumulate the mean of matrices
                self.model[label] += mat / len(trainX)

            # Compute rank k approximation of matrices with SVD
            for label, mat in self.model.items()
                U, S, V = np.linalg.svd(mat)
                self.model[label] = mat

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def predict(self, mat) -> str:
        closest_label = ""
        closest_dist = np.inf

        for label, label_mean in self.model.items():
            # Minimize the Frobenius norm of the difference in matrices
            dist = np.linalg.norm(mat - label_mean)

            if dist < closest_dist:
                closest_dist = dist
                closest_label = label

        return closest_label
