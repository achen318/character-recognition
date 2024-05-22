import json
import numpy as np

from models.base_model import BaseModel


class MeanValue(BaseModel):
    def __init__(self):
        super().__init__("mean_value.model")

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "r") as f:
                self.model = json.load(f)

        except FileNotFoundError:
            # Train the model
            for mat, char in zip(trainX, trainY):
                if char not in self.model:
                    self.model[char] = 0

                self.model[char] += mat.mean() / len(trainX)

            # Save the model
            with open(self.model_file, "w") as f:
                json.dump(self.model, f)

    def predict(self, mat) -> str:
        mean = mat.mean()

        closest_char = ""
        closest_diff = np.inf

        for char, char_mean in self.model.items():
            diff = abs(mean - char_mean)

            if diff < closest_diff:
                closest_diff = diff
                closest_char = char

        return closest_char
