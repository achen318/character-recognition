import pickle

import numpy as np

from models.base_model import BaseModel


class Bogus(BaseModel):
    def __init__(self):
        super().__init__("bogus.model")

    def predict(self, mat) -> int:
        return np.random.randint(0, 47)
