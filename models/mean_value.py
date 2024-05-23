import numpy as np
import pickle

from models.base_model import BaseModel


class MeanValue(BaseModel):
    def __init__(self):
        super().__init__("mean_value.model")

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)

        except FileNotFoundError:
            # Train the model
            for mat, label in zip(trainX, trainY):
                if label not in self.model:
                    self.model[label] = 0

                # Accumulate the mean of entries of the matrix
                self.model[label] += mat.mean() / len(trainX)

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def predict(self, mat) -> str:
        closest_label = ""
        closest_dist = np.inf

        for label, label_mean in self.model.items():
            # Minimize the absolute difference in mean values
            dist = abs(mat.mean() - label_mean)

            if dist < closest_dist:
                closest_dist = dist
                closest_label = label

        return closest_label
