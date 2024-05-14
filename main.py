import numpy as np
import random
from mnist import MNIST

mndata = MNIST("data")

images, labels = mndata.load_training()
# images, labels = mndata.load_testing()

X = np.array(random.choice(images)).reshape(28, 28)
# P, D, Q = np.linalg.svd(X, full_matrices=False)

print(X.view())
