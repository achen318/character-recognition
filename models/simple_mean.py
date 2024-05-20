import json
import math
import numpy as np

from models.base_model import BaseModel


class SimpleMean(BaseModel):
    def __init__(self):
        super().__init__()

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open("simple_mean.model", "r") as f:
                self.model = json.load(f)

        except FileNotFoundError:
            # Train the model
            for mat, char in zip(trainX, trainY):
                char = str(char)

                if char not in self.model:
                    self.model[char] = 0

                self.model[char] += mat.mean() / len(trainX)

            # Save the model
            with open("simple_mean.model", "w") as f:
                json.dump(self.model, f)

    def predict(self, mat) -> str:
        mean = mat.mean()

        closest_char = ""
        closest_diff = math.inf

        for char, char_mean in self.model.items():
            diff = abs(mean - char_mean)

            if diff < closest_diff:
                closest_diff = diff
                closest_char = char

        return closest_char
