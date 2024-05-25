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
                # Add image as a column vector to the model
                if label not in self.model:
                    self.model[label] = mat.reshape((784, 1))
                else:
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
            # Use the first 784 columns to ensure the matrix is 784x784
            label_span = label_span[:, :784]

            # Use least squares to predict the label
            m, c, _, _ = np.linalg.lstsq(label_span, flat_mat, rcond=None)
            p_label = m * flat_mat + c

            # Minimize the norm between p_label and flat_mat
            dist = np.linalg.norm(p_label - flat_mat)

            if dist < closest_dist:
                closest_dist = dist
                closest_label = label

        return closest_label
