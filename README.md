# character-recognition

A website demonstration of how linear algebra techniques (e.g. singular value decomposition) can be used to classify handwritten characters. An accompanying [slideshow](https://docs.google.com/presentation/d/1O5eeYbItA2KitufxAa5cZMYMJOYkmisl1KJQsGiyAv0/edit?usp=sharing) explains the mathematical intuition of the machine learning models developed for the project. The website is interactive, allowing users to draw a character and see the predictions of different models.

# Commands

## Install

1. `pip3 install -r requirements.txt`
2. `mkdir ~/.cache/emnist`
3. `wget https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip`
4. `mv gzip.zip ~/.cache/emnist/emnist.zip`

## Run

1. `python3 app.py`

## Clean

1. `rm *.model`

# Credits

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
