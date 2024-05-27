import pickle

import numpy as np

from models.base_model import BaseModel


class Bogus(BaseModel):
    def __init__(self):
        super().__init__("bogus.model")

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)

        except FileNotFoundError:
            # Train the model
            self.model = trainY

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def predict(self, mat) -> str:
        return np.random.choice(self.model)
