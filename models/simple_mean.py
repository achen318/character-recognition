import numpy as np


class SimpleMean:
    def __init__(self):
        self.means = {}

    def train(self, trainX, trainY):
        for image, digit in zip(trainX, trainY):
            if digit not in self.means:
                self.means[digit] = 0

            self.means[digit] += image.mean / len(trainX)
