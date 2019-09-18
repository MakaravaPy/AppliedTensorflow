Deep learning is a type of machine learning algorithm that uses Neural Networks to learn. The type of learning is "Deep" because it is composed of many layers of Neural Networks.

TensorFlow is a symbolic math Google's Library for Numerical Computation used for application such as neural networks. This library can do most of the things as numpy. However, it has a very different approach: instead of computing things immediately, we first define things that we want to compute later using what's called a Graph. Everything in Tensorflow takes place in a computational graph and running and evaluating anything in the graph requires a Session.

Note: This work is based on the [Kadenze Academy](https://www.kadenze.com/partners/kadenze-academy) course [Creative Applications of Deep Learning w/ Tensorflow.](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info)

## Tensorflow basics

Lets see a simple example of using tensorflow to download the data, compute mean, standard deviation and to normalize its.

```python
import tensorflow as tf
import os
 import numpy as np
 import matplotlib.pyplot as plt
 from skimage.transform import resize
dirname = ...  # your path to data
filenames = [os.path.join(dirname, fname) for fname in os.listdir(dirname)] 
imgs = [plt.imread(fname)[..., :3] for fname in filenames]
imgs = [utils.imcrop_tosquare(img_i) for img_i in imgs] # crop images to scquare 
imgs = [resize(img_i, (100, 100)) for img_i in imgs] #resize the square image to 100 x 100 pixels
imgs = np.array(imgs).astype(np.float32) # make list of 3-D images a 4-D array 
plt.figure(figsize=(10, 10)) # plot the resulting dataset
```

![[dataset.png|dataset.png]]

So, we have 100 pictures of bears. Now calculate the mean and standard deviation:

```python
sess = tf.Session() #create a tensorflow session
mean_img_op = tf.reduce_mean(imgs, axis=0) #create an operation that will calculate the mean of images
mean_img = sess.run(mean_img_op) #run that operation using session
mean_img_4d = tf.reduce_mean(imgs, axis=0, keepdims=True)#4 dimensional mean image
subtraction = imgs - mean_img_4d #compute the difference of every image with a 4 dimensional mean image
std_img_op = tf.sqrt(tf.reduce_mean(subtraction * subtraction, axis=0)) #compute the standard deviation
std_img = sess.run(std_img_op) #calculate the standard deviation using your session
sess.run(subtraction/std_img)
```

| ---------------------- | -------------------- |
|          Mean          |  Standard deviation  |
| ![[mean.png|mean.png]] | ![[std.png|std.png]] |

Now lets plot normalized dataset :

![[normalized1.png|normalized1.png]]

Tensorflow has a basic structure which is called placeholder. A placeholder is simply a variable that we will assign data to at a later date,  meaning, we're not sure what these are yet. It allows us to create our operations and build our computation graph, without needing the data, but we know they'll fit in the graph like so. Generally it is the input and output of the network. In TensorFlow terminology, we then feed data into the graph through these placeholders.  
Lets define a placeholder:

```python
X = tf.placeholder(tf.float32, name='X')
```

Whenever we create a neural network, we have to define a set of operations. These operations try to take us from some input to some output. For instance, the input might be an image, or frame of a video, or text file, or sound file. The operations of the network are meant to transform this input data into something meaningful that we want the network to learn about. Moreover, we want machine to find the best way to do this. The "best" in this case means, that we want to measure an error (difference) between what it should output and what it actually outputs: **E=f(X)-y**.

The main idea here is to see what happens to our error and use that knowledge to find smaller errors. Let's say the error went up after we moved our network parameters. Well then we know we should go back the way we came, and try going the other direction entirely. If our error went down, then we should just keep changing our parameters in the same direction. The error provides a "training signal" or a measure of the "loss" of our network.

That's known as a gradient descent. Gradient descent is a simple but very powerful method for finding smaller measures of error by following the negative direction of its gradient. In order to find the gradient we use the method known as backpropagation: When we pass as input something to a network, it's doing what's called forward propagation. We're sending an input **X** and multiplying it by every weight **W** to an expected output. Whatever differences that output has with the output we wanted it to have, gets backpropagated to every single parameter in our network.

### Cost function

The cost should represent how much error there is in the network, and provide the optimizer this value to help it train the network's parameters using gradient descent and backpropagation. Two most popular errors are **l~1~**~ ~(linear in error, by taking the absolute value of the error) and **l~2~**~  ~(known as least squares).  
Example of l~2 ~error:

```python
error = tf.squared_difference(Y_pred,Y) #compute the distance
sum_error = tf.reduce_sum(error,1) # sum the error for each feature in Y
cost = tf.reduce_mean(sum_error) # compute the cost. as the mean error of the batch
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) #optimizer with learning_rate says how far along the gradient to move
```

### Fully Connected Network

One more important formula we need for connecting an input to a network to a series of fully connected (linear) layers: **H=f(XW+b)**. Here, **H** is an output of a hidden layer representing the "hidden" activations of a network, **f** represents some nonlinearity, **X** represents input to that layes, W is that weight matrix, and **b** is that layer's bias. The bias **b** allows for a global shift in the resulting values. Nonlinearity of **f** allows the input space to be nonlinearly warped, allowing it to express a lot more interesting distributions of data. The weights **W **will be updated by optimizer. Summurize this in the following function.

```python
def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(name='W',shape=[n_input,n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(name='b',shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h
```

## Training  a network

Now we teach our network to go from the location on an image frame to a particular color. So given any position in an image, the network will need to learn what color to paint. For this we should aggregate pixel locations and their colors and normolized the input values. What we're going to do is use regression to predict the value of a pixel given its (row, col) position. So the input to our network is X = (row, col) value. And the output of the network is Y = (r, g, b). But when we give inputs of (row, col) to our network, it won't know what order they are, because we will randomize them. So it will have to learn what color value should be output for any given (row, col). Now create a deep neural network that takes your network input X of 2 neurons, multiplies it by a linear and non-linear transformation which makes its shape [None, 20], meaning it will have 20 output neurons. Then repeat the same process again to give you 20 neurons again, and then again and again until you've done 6 layers of 20 neurons. Then finally one last layer which will output 3 neurons, your predicted output for a total of 6 hidden layers, or 8 layers total including the input and output layers.

Consider our reference picture and gif results what we get after 500 interpolations:

| -------------------------------- | ---------------------------- |
|        Reference picture         |            Result            |
| ![[reference.png|reference.png]] | ![[single1.gif|single1.gif]] |

Now we apply the same approach to paint every single image in our bear-dataset with 100 images.  In order to find the best way now we could for instance feed in every possible image by having multiple row, col -> r, g, b values. So for any given row, col, we'd have 100 possible r, g, b values. This likely won't work very well as there are many possible values a pixel could take, not just one.

![[multiple1.gif|multiple1.gif]]

What we're seeing is the training process over time. We feed in our input positios, which consist of the pixel values of each of our 100 images, it goes through the neural network, and out come predicted color values for every possible input value. We visualize it above as a gif by seeing how at each iteration the network has predicted the entire space of the inputs. We can visualize just the last iteration as a "latent" space, going from the first image (the top left image in the montage), to the last image, (the bottom right image).

![[final1.gif|final1.gif]]

## Unsupervised and Supervised Learning

Machine learning research in deep networks performs one of two types of learning. You either have a lot of data and you want the computer to reason about it, maybe to encode the data using less data, and just explore what patterns there might be. That's useful for clustering data, reducing the dimensionality of the data, or even for generating new data. That's generally known as unsupervised learning. In the supervised case, you actually know what you want out of your data. You have something like a label or a class that is paired with every single piece of data. We see how unsupervised learning works using something called an autoencoder and how it can be extended using convolution. Then we get into supervised learning and show how we can build networks for performing regression and classification.

### Autoencoder

An autoencoder is a type of neural network that learns to encode its inputs, often using much less data. It does so in a way that it can still output the original input with just the encoded values. For it to learn, it does not require "labels" as its output. Instead, it tries to output whatever it was given as input. So in goes an image, and out should also go the same image. But it has to be able to retain all the details of the image, even after possibly reducing the information down to just a few numbers.

We write two functions to preprocess (normalize) any given image, and to unprocess it, i.e. unnormalize it by removing the normalization. The preprocess function subtracts the mean, then divides by the standard deviation. The deprocess function should take the preprocessed image and undo the preprocessing steps. Recall that the ds object contains the mean and std functions for access the mean and standarad deviation. We use the preprocess and deprocess functions on the input and outputs of the network.

```python
def preprocess(img, ds):
    norm_img = (img - np.mean(img))/np.std(img)
    return norm_img
def deprocess(norm_img, ds):
    img = norm_img * np.std(norm_img) +np.mean(norm_img)
    return img
```

Let's have a look what the network learns to encode first, based on what it is able to reconstruct. It won't able to reconstruct everything. At first, it will just be the mean image. Then, other major changes in the dataset. From the basic interpretation, you can reason that the autoencoder has learned a representation of the backgrounds, and is able to encode that knowledge of the background in its inner most layer of just two values. It then goes on to represent the major variations in bear position. So the features it is able to encode tend to be the major things at first, then the smaller things. With 2000 (with bigger number I run into memoty problems) iterations we receive the following reconstructed picture.

| ---------------------- | ------------------------ |
|    Original picture    |  Reconstructed picture   |
| ![[test.png|test.png]] | ![[recon.png|recon.png]] |

### Convolutional Autoencoder

To get even better encodings, we can also try building a convolutional network. Why would a convolutional network perform any different to a fully connected one? In the fully connected network for every pixel in our input, we have a set of weights corresponding to every output neuron. Those weights are unique to each pixel. Each pixel gets its own row in the weight matrix. That really doesn't make a lot of sense, since we would guess that nearby pixels are probably not going to be so different. And we're not really encoding what's happening around that pixel, just what that one pixel is doing.

In a convolutional model, we're explicitly modeling what happens around a pixel. And we're using the exact same convolutions no matter where in the image we are. But we're going to use a lot of different convolutions. Let see the results of latent manifold and  the reconstruction created by the encoder/variational encoder.

| ------------------------------ | ---------------------------------------- |
| ![[Manifold.gif|Manifold.gif]] | ![[Reconstructed.gif|Reconstructed.gif]] |

### Denoising Autoencoder[�](http://localhost:8888/notebooks/CADL/session-3/lecture-3.ipynb#Denoising-Autoencoder)

The denoising autoencoder is a very simple extension to an autoencoder. Instead of seeing the input, it is corrupted, for instance by masked noise. but the reconstruction loss is still measured on the original uncorrupted image. What this does is lets the model try to interpret occluded or missing parts of the thing it is reasoning about. It would make sense for many models, that not every datapoint in an input is necessary to understand what is going on. Denoising autoencoders try to enforce that, and as a result, the encodings at the middle most layer are often far more representative of the actual classes of different objects.