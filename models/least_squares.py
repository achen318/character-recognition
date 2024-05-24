import matplotlib.pyplot as plt
import numpy as np
import pickle

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
            # Train the model
            for mat, label in zip(trainX, trainY):
                if label not in self.model:
                    self.model[label] = np.zeros((784, 1))

                # Add image as a column vector to the model
                self.model[label] = np.hstack(
                    (self.model[label], mat.reshape((784, 1)))
                )

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

        flat_mat = mat.reshape((784, 1))

        for label, label_span in self.model.items():
            # Minimize the Frobenius norm of the difference in matrices
            dist = np.linalg.norm(flat_mat - label_span)

            if dist < closest_dist:
                closest_dist = dist
                closest_label = label

        return closest_label
