import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from models.base_model import BaseModel


class NeuralNetwork(BaseModel):
    def __init__(self, layers: List[int], iterations: int, alpha: int):
        super().__init__("neural_network.model")
        self.layers = layers # [784, hiddens, 10]
        self.iterations = iterations
        self.alpha = alpha # learning rate

    def train(self, trainX, trainY) -> None:
        try:
            # If the model exists, load it
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)

        except FileNotFoundError:
            # Initialize parameters
            self.model = []

            num_layers = len(self.layers)
            num_params = num_layers - 1
            num_data = trainX.shape[0]

            for i in range(num_params):
                W = np.random.randn(self.layers[i], self.layers[i + 1])
                B = np.random.randn(num_data, self.layers[i + 1])

                self.model.append([W, B])
            
            # Train the model
            for i in range(self.iterations):
                # ----- Forward propagation -----
                A = [None] * num_params
                Z = [None] * num_params

                for j in range(num_params):
                    W_j, B_j = self.model[j]

                    # Z = WA + B
                    if j == 0:
                        Z[j] = trainX.T @ W_j + B_j
                    else:
                        Z[j] = A[j - 1] @ W_j + B_j

                    # A = activiation(Z)
                    if j < num_params - 1:
                        A[j] = np.maximum(0, Z[j]) # A = ReLU(Z)
                    else:
                        A[j] = np.exp(Z[j]) / np.sum(np.exp(Z[j])) # A = Softmax(Z)

                # ----- Backward propagation -----
                dZ = [None] * num_params
                dW = [None] * num_params
                dB = [None] * num_params

                for j in range(num_params -1, 0, -1):
                    if j == num_params - 1:
                        dZ[j] = A[j] - trainY.T
                    else:
                        dZ[j] = self.model[j][0].T @ dZ[j]

                    dW[j] = dZ[j] @ A[j-1].T / num_data
                    dB[j] = np.sum(dZ[j]) / num_data

                # ----- Update parameters -----
                for j in range(num_params):
                    self.model[j][0] -= self.alpha * dW[j]
                    self.model[j][1] -= self.alpha * dB[j]

                # Save the model every 10 iterations
                if i % 10 == 0:
                    print(f"Iteration {i}")

                    with open(self.model_file, "wb") as f:
                        pickle.dump(self.model, f)

            # Save the model
            with open(self.model_file, "wb") as f:
                pickle.dump(self.model, f)

    def display(self) -> None:
        # Show the coefficients/weights matrix
        plt.imshow(self.model.reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.show()

    def predict(self, mat) -> str:
        X = mat.reshape(1, -1)
        Y = X @ self.model  # returns a decimal

        return round(Y[0])
