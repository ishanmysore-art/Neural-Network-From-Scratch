# Neural Network From Scratch

A fully connected feedforward neural network implemented entirely from scratch in Java. This project recreates the core mechanics behind modern deep learning frameworks by implementing forward propagation, backpropagation, gradient descent, and customizable network architectures without relying on external machine learning libraries.

---

## Overview

The goal of this project is to understand how neural networks work at a fundamental level by building every component from first principles.

Rather than using libraries such as TensorFlow or PyTorch, this implementation performs all computations manually, including matrix operations, activation functions, gradient calculations, and parameter updates.

The network supports supervised learning through backpropagation and gradient descent, allowing it to learn nonlinear relationships directly from data.

---

## Features

### Feedforward Neural Network
- Fully connected (dense) architecture
- Configurable number of layers
- Customizable neurons per layer
- Support for multiple hidden layers

### Forward Propagation
- Manual computation of weighted sums
- Bias handling
- Layer-by-layer activation computation

### Backpropagation
- Gradient computation using the chain rule
- Error propagation through hidden layers
- Automatic weight and bias updates

### Optimization
- Gradient descent training
- Configurable learning rate
- Iterative parameter optimization

### Activation Functions
Implemented common nonlinear activation functions, including:
- Sigmoid
- Tanh

---

## Learning Pipeline

```
Training Data
      │
      ▼
Forward Propagation
      │
      ▼
Prediction
      │
      ▼
Loss Computation
      │
      ▼
Backpropagation
      │
      ▼
Gradient Descent
      │
      ▼
Updated Parameters
```

---

## Architecture

```
Input Layer
      │
      ▼
Hidden Layer(s)
      │
      ▼
Output Layer
```

Each neuron computes

```
z = Wx + b
a = activation(z)
```

During training, gradients are computed using backpropagation and used to update weights:

```
W = W − α∇W
b = b − α∇b
```

where:

- **α** = learning rate
- **∇W** = weight gradients
- **∇b** = bias gradients

---

## Project Structure

```
src/
├── NeuralNetwork.java
├── Layer.java
├── Neuron.java
├── Matrix.java
├── ActivationFunctions.java
├── LossFunctions.java
├── Trainer.java
└── Main.java
```

*(Directory names may differ depending on your implementation.)*

---

## Technologies

- Java
- Object-Oriented Programming
- Linear Algebra
- Calculus
- Gradient Descent
- Backpropagation
- Machine Learning Fundamentals

---

## Key Concepts Explored

This project demonstrates understanding of:

- Artificial Neural Networks
- Feedforward computation
- Matrix mathematics
- Activation functions
- Cost functions
- Gradient descent optimization
- Backpropagation
- Weight initialization
- Bias learning

---

## Why Build a Neural Network From Scratch?

Machine learning libraries abstract away much of the underlying mathematics. This project focuses on implementing the algorithms manually to gain a deeper understanding of how neural networks actually learn.

Building the network from first principles provides insight into:

- How predictions are computed
- How gradients are derived
- Why backpropagation works
- How optimization improves model performance

---

## Future Improvements

Potential extensions include:

- Mini-batch gradient descent
- Stochastic gradient descent (SGD)
- Adam optimizer
- Softmax output layer
- Cross-entropy loss
- Dropout regularization
- Batch normalization
- Model serialization
- GPU acceleration
- Convolutional neural networks (CNNs)

---

## Author

**Ishan Mysore**
