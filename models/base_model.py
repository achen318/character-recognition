from typing import Tuple


class BaseModel:
    def __init__(self):
        self.model = {}

    def train(self, trainX, trainY) -> None: ...

    def predict(self, mat) -> str: ...

    def test(self, testX, testY) -> Tuple[int, int]:
        correct = 0
        total = 0

        for mat, char in zip(testX, testY):
            if self.predict(mat) == str(char):
                correct += 1

            total += 1

        return (correct, total)
