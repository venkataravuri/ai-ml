# Neural Networks & Deep Learning

## Pytorch

### Tensor Operations

Tensor Operations <a target="_blank" href="https://colab.research.google.com/github/venkataravuri/ai-ml-models/blob/main/3-deeplearning-with-pytorch/1-basics/1-tensor-operations.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Artificial Neuron / Perceptron

<img src="https://miro.medium.com/v2/resize:fit:640/1*sPg-0hha7o3iNPjY4n-vow.jpeg" width="40%" height="40%" />

#### Forward Pass

Forward pass computes network output and “error”

Input data -> Neural Network -> Prediction

#### Backpropagation (Backward pass)

- Backward pass to compute gradients
- A fraction of the weight’s gradient is subtracted from the weight. (based on learning rate)

Neural Network <- Measure of Error

Adujst to reduce error

Learning is an Optimization Problem

-Update the weights and biases to decrease loss function

- [Understanding Backpropagation - A visual derivation of the equations that allow neural networks to learn](https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0)

A Neural Network introductory videos with nice intutions. A 4-part series that explaines neural networks very intutively.

- :tv: [What is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- :tv: [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)
- :tv: [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- :tv: [Backpropagation calculus | Chapter 4, Deep learning](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)


## Neural Networks

- [Please Stop Drawing Neural Networks Wrong](https://towardsdatascience.com/please-stop-drawing-neural-networks-wrong-ffd02b67ad77)

## Deep Learning

Deep learning is a type of neural networks with multiple layers which can handle higher-level computation tasks, such as natural language processing, fraud detection, autonomous vehicle driving, and image recognition.

Deep learning models and their neural networks include the following:
- Convolution neural network (CNN).
- Recurrent neural network (RNN).
- Generative adversarial network (GAN).
- Autoencoder.
- Generative pre-trained transformer (GPT).

Deep Learning modeling is still an art, in other words, intelligent trial and error. There is no universal answer for NNs.

### NN Vs. RNN

Fundamental difference in the preservation of the network state, which is absent in the feed-forward network.

#### Generative Adversarial Network (GAN)

GANs are a way to make generative model by having two neural network models compete each other.

- [Face Generation Using Generative Adversarial Networks (GAN)](https://medium.com/nerd-for-tech/face-generation-using-generative-adversarial-networks-gan-6d279c2d5759)

- [Deep Convolutional Generative Adversarial Networks(DCGANs)](https://medium.datadriveninvestor.com/deep-convolutional-generative-adversarial-networks-dcgans-3176238b5a3d)

#### Diffusions



|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:tv:|[MIT Deep Learning Course - Lex Fridman](https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf)

#### CNN

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:tv:|[Nice intution about CNN, Convolutional Neural Networks](https://www.youtube.com/watch?v=xg2ajb3csgk&list=PLXAoLgwZtKcgGE2-Wy23EUE4Q03s-YVwF&index=3)|

#### RNN

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:tv:|[A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg)

### Activation Functions

Activation functions transforms the weighted sum of a neuron so that the output is non-linear

#### Sigmoid

The most common sigmoid function used in machine learning is Logistic Function, as the formula below.

<img src="https://miro.medium.com/v2/resize:fit:382/format:webp/1*VE7kb7J2lEo5zsUGyHgMWQ.png" />

It has following properties:
* It maps the feature space into probability functions
* It uses exponential
* It is differentiable

Sigmoid function has values that are extremely near to 0 or 1. Because of this, it is appropriate for use in **binary classification problems**.

#### Softmax

The Softmax function is a generalized form of the logistic function.

Softmax function has following characterstics and widely used in multi-class classification problems:
* It maps the feature space into probability functions
* It uses exponential
* It is differentiable

Softmax is used for multi-classification. The probabilities sum will be 1

References: 
* https://towardsdatascience.com/understanding-sigmoid-logistic-softmax-functions-and-cross-entropy-loss-log-loss-dbbbe0a17efb

### Loss Vs. Optimizer

Think of loss function has what to minimize and optimizer how to minimize the loss.

The *loss* is way of measuring difference between target label(s) and prediction label(s).
* Loss could be "mean squared error", "mean absolute error loss also known as L2 Loss", "Cross Entropy" ... and in order to reduce it, weights and biases are updated after each epoch. Optimizer is used to calculate and update them.

The optimization strategies aim at minimizing the cost function.

#### Loss Function

Loss function quantifies gap between prediction and ground truth.

For regression:
- Mean Squared Error (MSE)

For classification:
- Cross Entropy Loss

##### Loss Function Vs. Cost Function

Cost function and loss function are synonymous and used interchangeably, they are different.

A loss function is for a single training example. It is also sometimes called an error function. A cost function, on the other hand, is the average loss over the entire training dataset.

the log magnifies the mistake in the classification, so the misclassification will be penalized much more heavily compared to any linear loss functions. The closer the predicted value is to the opposite of the true value, the higher the loss will be, which will eventually become infinity. That’s exactly what we want a loss function to be.

#### Cross Entropy Loss

when designing a neural network multi-class classifier, you can you CrossEntropyLoss with no activation, or you can use NLLLoss with log-SoftMax activation.

#### Underfitting, overfitting, and generalization

A good classifier will learn a decision boundary that correctly classifies most of the training data and generalizes to novel data.

Overfitting: The error decreases in the training set but increases in the test set. Overfitting example (a sine curve vs 9-degree polynomial)

Regularization


Most common regularization methods are,

1) Early Stopping - Stop training (or at least save a checkpoint) when performance on the validation set decreases. Monitor model performance on a validation set and stop training when performance degrades.

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*b4lf4K0PswVYZXdI.png" />

3) Dropout - Randomly remove some nodes in the network (along with incoming and outgoing edges). Probabilistically remove nodes during training.
- Usually p >= 0.5 (p is probability of keeping node)
- Input layers p should be much higher (and use noise instead of dropout)
- Most deep learning frameworks come with a dropout layer

- Dropout is not used after training when making a prediction with the fit network.

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*YCofAkhSErYvlpRT.png" />

3) Batch Normalization (BatchNorm, BN)
- Normalize hidden layer inputs to mini-batch mean & variance
- Reduces impact of earlier layers on later layers


4) Weight Penalty (aka Weight Decay) - The simplest and perhaps most common regularization method is to add a penalty to the loss function in proportion to the size of the weights in the model. Penalize the model during training based on the magnitude of the weights.

5) Weight Constraint: Constrain the magnitude of weights to be within a range or below a limit.

6) Activity Regularization: Penalize the model during training base on the magnitude of the activations.

7) Noise: Add statistical noise to inputs during training.

### Vanshing Gradient Problem

Neural networks are trained by gradient descent. TDuring neural network training, stochastic gradient descent (SGD) computes the gradient of the loss functions w.r.t. its weights in the network. However, in some cases, this gradient becomes exceedingly small – hence the term “vanishing gradient.”

SGD updates a weight based on its gradient, the update is proportional to that gradient. If the gradient is vanishingly small, then the update will be vanishingly small too.

Due to this, the network learns very slowly. Also, if the adjustments are too small, the network can get stuck in a rut and stop improving altogether. 

How to identify if your model suffers from a vanishing gradient problem:

- If the loss decreases very slowly over time while training.
- If you observe erratic behavior or diverging training curves.
- If the gradients in the early layer of the network are becoming increasingly small while propagating backward.

Residual connections mainly help mitigate the vanishing gradient problem.

### Residual Connections in NN

Residual connections mainly help mitigate the vanishing gradient problem. During the back-propagation, the signal gets multiplied by the derivative of the activation function. In the case of ReLU, it means that in approximately half of the cases, the gradient is zero. Without the residual connections, a large part of the training signal would get lost during back-propagation. Residual connections reduce effect because summation is linear with respect to derivative, so each residual block also gets a signal that is not affected by the vanishing gradient. The summation operations of residual connections form a path in the computation graphs where the gradient does not get lost.

Another effect of residual connections is that the information stays local in the Transformer layer stack. The self-attention mechanism allows an arbitrary information flow in the network and thus arbitrary permuting the input tokens. The residual connections, however, always "remind" the representation of what the original state was. To some extent, the residual connections give a guarantee that contextual representations of the input tokens really represent the tokens.


# PyTorch

## Pytorch JIT

A deep learning framework is said to use eager execution (or eager evaluation) if it builds its computational graph (the set of steps needed to perform forward or backwards propagation through the network) at runtime. PyTorch is the classic example of a framework which is eagerly evaluated. Every forward pass through a PyTorch model constructs an autograd computational graph; the subsequent call to backwards then consumes (and destroys!) this graph (for more on PyTorch autograd, see here).

Constructing and deconstructing objects in this way paves the way to a good developer experience. The code that’s actually executing the mathematical operations involved is ultimately a C++ or CUDA kernel, but the result of each individual operation is immediately transferred to (and accessible from) the Python process, because the Python process is the “intelligent” part—it’s what’s managing the computational graph as a whole. This is why debugging in PyTorch is as simple as dropping import pdb; pdb.set_trace() in the middle of your code.

Alternatively, one may make use of graph execution. Graph execution pushes the management of the computational graph down to the kernel level (e.g. to a C++ process) by adding an additional compilation step to the process. The state is not surfaced back to the Python process until after execution is complete.


# References

- [Deep Learning Course](https://fleuret.org/dlc/?utm_campaign=Data_Elixir&utm_source=Data_Elixir_458)
https://www.dropbox.com/s/c0g3sc1shi63x3q/deep_learning_basics.pdf?dl=0

