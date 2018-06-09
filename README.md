# Deep Learning with docker and jupyter notebooks

This forked repository contains Jupyter notebooks implementing the code samples found in the book  
[Deep Learning with Python (Manning Publications)](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff).

## Table of contents

* Chapter 2:
    * [2.1: A first look at a neural network](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/2.1-a-first-look-at-a-neural-network.ipynb)
* Chapter 3:
    * [3.5: Classifying movie reviews](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/3.5-classifying-movie-reviews.ipynb)
    * [3.6: Classifying newswires](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/3.6-classifying-newswires.ipynb)
    * [3.7: Predicting house prices](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/3.7-predicting-house-prices.ipynb)
* Chapter 4:
    * [4.4: Underfitting and overfitting](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/4.4-overfitting-and-underfitting.ipynb)
* Chapter 5:
    * [5.1: Introduction to convnets](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/5.1-introduction-to-convnets.ipynb)
    * [5.2: Using convnets with small datasets](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/5.2-using-convnets-with-small-datasets.ipynb)
    * [5.3: Using a pre-trained convnet](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/5.3-using-a-pretrained-convnet.ipynb)
    * [5.4: Visualizing what convnets learn](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/5.4-visualizing-what-convnets-learn.ipynb)
* Chapter 6:
    * [6.1: One-hot encoding of words or characters](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/6.1-one-hot-encoding-of-words-or-characters.ipynb)
    * [6.1: Using word embeddings](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/6.1-using-word-embeddings.ipynb)
    * [6.2: Understanding RNNs](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/6.2-understanding-recurrent-neural-networks.ipynb)
    * [6.3: Advanced usage of RNNs](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/6.3-advanced-usage-of-recurrent-neural-networks.ipynb)
    * [6.4: Sequence processing with convnets](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/6.4-sequence-processing-with-convnets.ipynb)
* Chapter 8:
    * [8.1: Text generation with LSTM](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/8.1-text-generation-with-lstm.ipynb)
    * [8.2: Deep dream](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/8.2-deep-dream.ipynb)
    * [8.3: Neural style transfer](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/8.3-neural-style-transfer.ipynb)
    * [8.4: Generating images with VAEs](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/8.4-generating-images-with-vaes.ipynb)
    * [8.5: Introduction to GANs](http://nbviewer.jupyter.org/github/lorosanu/deep-learning-with-python-notebooks/blob/master/notebooks/8.5-introduction-to-gans.ipynb
)

## Running notebooks on CPU

* Install dependencies
  * Git
  * [Docker Community Edition](https://www.docker.com/community-edition#/download)

* Clone this repository

    ```
    $ git clone https://github.com/lorosanu/deep-learning-with-python-notebooks
    $ cd deep-learning-with-python-notebooks
    ```

* Use docker-compose to build and use the image

    ```
    $ docker-compose up

      Copy/paste this URL into your browser when you connect for the first time,
      to login with a token:
        http://0.0.0.0:8888/?token=6a99...
    ```

* The jupyter notebook will be available at `http://0.0.0.0:8888/?token=6a99...`

* __Note__: Mac and Windown users will have to find the IP address of docker-machine VM and replace 'localhost' with it.  
More [here](https://docs.docker.com/docker-for-windows/troubleshoot/#limitations-of-windows-containers-for-localhost-and-published-ports).
