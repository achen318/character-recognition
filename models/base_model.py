from typing import Tuple


class BaseModel:
    def __init__(self, model_file):
        self.model = {}
        self.model_file = model_file

    def train(self, trainX, trainY) -> None: ...

    def display(self) -> None: ...

    def predict(self, mat) -> str: ...

    def test(self, testX, testY) -> Tuple[int, int]:
        correct = 0
        total = 0

        for mat, label in zip(testX, testY):
            if self.predict(mat) == label:
                correct += 1

            total += 1

        return (correct, total)
