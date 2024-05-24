import matplotlib.pyplot as plt
import numpy as np
import pickle

from models.base_model import BaseModel


class SVD(BaseModel):
    def __init__(self, k: int):
        super().__init__(f"svd_{k}.model")
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

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def display(self) -> None:
        keys = sorted(self.model.keys())

        for i, label in enumerate(keys):
            plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
            plt.imshow(self.model[label], cmap="gray")
            plt.axis("off")
            plt.title(label)

        plt.show()

    def predict(self, mat) -> str:
        closest_label = ""
        closest_dist = np.inf

        for label, label_mean in self.model.items():
            # Perform SVD on the difference in matrices
            U, S, V = np.linalg.svd(mat - label_mean)

            # Minimize the Frobenius norm of the rank k approximation
            dist = np.linalg.norm(
                U[:, : self.k] @ np.diag(S[: self.k]) @ V[: self.k, :]
            )

            if dist < closest_dist:
                closest_dist = dist
                closest_label = label

        return closest_label
