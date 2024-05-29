# character-recognition

A website demonstration of how linear algebra techniques (e.g. singular value decomposition) can be used to classify handwritten characters. An accompanying [slideshow](https://docs.google.com/presentation/d/1O5eeYbItA2KitufxAa5cZMYMJOYkmisl1KJQsGiyAv0/edit?usp=sharing) explains the mathematical intuition of the machine learning models developed for the project. The website is interactive, allowing users to draw a character and see the predictions of different models.

# Commands

| Install                                                     | Run              | Clean            |
| ----------------------------------------------------------- | ---------------- | ---------------- |
| `pip3 install -r requirements.txt`                          | `python3 app.py` | `rm -rf *.model` |
| `mkdir ~/.cache/emnist`                                     |                  |                  |
| `wget https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip` |                  |                  |
| `mv gzip.zip ~/.cache/emnist/emnist.zip`                    |                  |                  |

# Credits

- Linear Algebra with Mr. Honner

- [Naoki Saito (MAT 167 @ UC Davis) - Lecture 21: Classification of Handwritten Digits](https://www.math.ucdavis.edu/~saito/courses/167.s12/Lecture21.pdf)

- [MachineLearningMastery - CNN for MNIST](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)

- [3Blue1Brown (YouTube) - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

- [Samson Zhang (YouTube) - Neural Network from Scratch](https://youtu.be/w8yWXqWQYmU)

- [Azka Redhia (Medium) - Neural Network for Handwritten Digit Recognition](https://medium.com/@azkardm/handwritten-digit-recognition-4dc904edb515)

- [Kaggle - EMNIST (Extended MNIST)](https://www.kaggle.com/datasets/crawford/emnist)

- [Achintya Tripathi (Kaggle) - 97.9% accurate EMNIST code](https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78/notebook)

- Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
