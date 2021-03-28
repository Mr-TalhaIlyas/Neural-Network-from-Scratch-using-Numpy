[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

<img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />
# Neural Network form scratch (MNIST)
This Repo creates a Neural Network form scratch using only **numpy** library. The NN is tested on MNIST-Digits dataset which contains 10 classes for classification.
MNIST like a 'hello_world' dataset in machine learning community.
This repo also builds the popular optimizers like;
* *ADAM*
* *RMSprop*
* *Adagrad*
* *Gradient descent with and without momentum term*
by using only numpy
The effect of different activations functions (e.g. *sigmoid*, *ReLu*) is also studied in this repo.

This repo takes you through the steps regarding what is happening under the hood of artificial neural networks created by high level libraries like
* Tenforflow
* Keras
* Pytorch

As this repo only uses **numpy** so it only runs on CPU that's why its considerably slower than the models running on GPUs.

### Dependencies
1. numpy
2. tqdm
3. scipy
4. matlab
5. idx2numpy

# Building the Neural Network

If you are completely new to machine learning I recommend you to read this [book](https://b-ok.asia/book/2701405/13985c) along side this repo.
I'll be using a lot of concepts discussesd in the book above and also in [cs231](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv).

#### Neural Network Architecture
## Training with *Sigmoid* activation
Follwing is the architecture of the neural network constructed in the scripts.

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/NN.gif?raw=true)

*You can study the details and math behind the neural network in the book above, here I'll explain things via presentation style images. Following is actually a **summary** of what you'll study in above mentioned lectures*

Let's first look at a single neuron(with sigmoid)

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img3.jpg?raw=true)

Now, let's look at sigmoid function and how to impliment in python.

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img2.jpg?raw=true)

Now we can write the code for forward propagation.

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img4.jpg?raw=true)

The following images summarize how the backpropagation works in neural networks and how the loss function back propagates the loss and how the gradients are calculated.

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img6.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img5.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img7.jpg?raw=true)

Finally now we can update the weights and biases via following equations;
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img8.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img9.jpg?raw=true)

This image shows the role of optimizers in updataing weights and biases, following images show how the weights are updated using momentum. *If you want to use any other optimizer like **ADAM, RMSprop or Adagrad** you just have to update the following equations.*
The scripts in this repo implement differnt optimizers.
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img10.jpg?raw=true)

## Training wiht *ReLu* activation

Another activation function that is still used in  many popular CNN architecture is RelU, it is implemented as;
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img12.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img13.jpg?raw=true)

and forward and bak prop work as follows

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img14.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img15.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img16.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img17.jpg?raw=true)

Following images compare the results of training the network with different activations

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img18.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img19.jpg?raw=true)

## Detailed Results

**Descriptions are provided in each slide**

![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img31.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img32.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img33.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img34.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img41.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img42.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img43.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img44.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Neural-Network-from-Scratch-using-Numpy/blob/master/screens/img45.jpg?raw=true)












