# N-Layer Neural Network From Scratch

A configurable feedforward neural network implemented entirely from scratch in Java. This project supports arbitrary network architectures, forward propagation, backpropagation, gradient descent training, configurable runtime modes, and binary weight serialization—all without using external machine learning libraries.

---

## Overview

This project explores the core algorithms behind modern neural networks by implementing every component from first principles.

Unlike frameworks such as TensorFlow or PyTorch, every stage of the learning process is written manually, including:

- Network configuration
- Weight initialization
- Forward propagation
- Error computation
- Backpropagation
- Gradient descent optimization
- Model persistence
- Inference

The implementation is generalized to support **any number of layers**, making it significantly more flexible than a traditional single-hidden-layer educational neural network.

---

## Features

### Configurable N-Layer Architecture

The network is completely configurable through a control file.

Users can specify:

- Number of layers
- Neurons per layer
- Learning rate
- Maximum training iterations
- Error threshold
- Weight initialization strategy
- Runtime mode (TRAIN or RUN)
- Input and truth datasets

Rather than hardcoding a fixed architecture, the implementation dynamically allocates memory for arbitrary feedforward networks.

Example architectures:

```
2 → 2 → 1
```

```
4 → 8 → 8 → 2
```

```
784 → 128 → 64 → 10
```

---

### Forward Propagation

Implements manual feedforward computation through each layer of the network.

For every neuron, the network computes:

```
Weighted Sum
      ↓
Activation Function
      ↓
Output Activation
```

The implementation performs every calculation using Java arrays and nested loops without relying on external numerical libraries.

---

### Backpropagation

Implements the complete backpropagation algorithm from scratch.

Training includes:

- Output error computation
- Hidden-layer error propagation
- Gradient computation
- Weight updates using gradient descent

The implementation supports networks with any number of hidden layers through generalized recursive computations.

---

### Gradient Descent Optimization

Weights are updated after each training example using gradient descent.

Training stops automatically when either:

- The average error falls below a user-defined threshold
- The maximum number of iterations is reached

Progress can optionally be reported at configurable intervals.

---

### Activation Functions

Implemented:

- Sigmoid
- Hyperbolic Tangent (tanh)

The current implementation uses the sigmoid activation function during training and inference while providing an extensible framework for additional activation functions.

---

### Flexible Weight Initialization

Supports three initialization modes:

- Manual initialization
- Random initialization
- Load existing weights from file

This allows experiments to be reproduced and previously trained networks to be reused without retraining.

---

### Model Persistence

The network can save trained weights directly to a binary file and reload them later.

Features include:

- Binary serialization
- Configuration validation before loading
- Reproducible inference
- Continued training from saved checkpoints

---

### Dataset Support

The network accepts datasets from multiple sources.

Supported formats include:

- Plain-text input tables
- Plain-text truth tables
- Binary activation files stored in folders

This allows the same implementation to be used on datasets beyond simple Boolean logic problems.

---

### Runtime Modes

The program supports two independent execution modes.

#### TRAIN

- Load training data
- Initialize or load weights
- Train the network
- Report convergence statistics
- Optionally save trained weights

#### RUN

- Load trained weights
- Execute forward propagation
- Produce predictions without retraining

---

## Compiler Architecture

```
                 Configuration File
                        │
                        ▼
             Load Network Parameters
                        │
                        ▼
              Allocate Network Memory
                        │
                        ▼
          Initialize or Load Weights
                        │
                        ▼
          ┌────────────────────────┐
          │        TRAIN           │
          │                        │
          │ Forward Propagation    │
          │        ↓               │
          │ Error Computation      │
          │        ↓               │
          │ Backpropagation        │
          │        ↓               │
          │ Gradient Descent       │
          └────────────────────────┘
                        │
                        ▼
               Save Weights (Optional)
                        │
                        ▼
               Forward Propagation
                        │
                        ▼
                  Network Outputs
```

---

## Example Applications

The implementation demonstrates supervised learning on standard Boolean logic problems, including:

- AND
- OR
- XOR

Because the architecture is configurable, the same implementation can be adapted to larger supervised learning datasets by modifying the configuration and input files.

---

## Technologies

- Java
- Object-Oriented Programming
- Feedforward Neural Networks
- Backpropagation
- Gradient Descent
- Binary File I/O
- Recursive Algorithms
- Numerical Computing

---

## Project Structure

```
NLayer.java
├── Configuration loading
├── Memory allocation
├── Weight initialization
├── Dataset loading
├── Forward propagation
├── Backpropagation
├── Gradient descent training
├── Weight serialization
├── Inference
└── Reporting
```

---

## Key Engineering Concepts

This project demonstrates experience with:

- Neural network implementation from first principles
- Configurable software architecture
- Dynamic memory allocation
- Recursive feedforward computation
- Recursive backpropagation
- Numerical optimization
- Binary serialization
- File parsing
- Object-oriented software design
- Algorithm implementation in Java

---

## Future Improvements

Potential extensions include:

- Mini-batch gradient descent
- Stochastic Gradient Descent (SGD)
- Adam optimizer
- Softmax output layer
- Cross-entropy loss
- Bias parameters
- Additional activation functions
- Regularization techniques
- Automatic dataset normalization
- GPU acceleration

---

## Why This Project?

Most machine learning libraries abstract away the mechanics of learning.

This project rebuilds those algorithms from the ground up, providing a deeper understanding of how neural networks compute predictions, propagate error signals, and iteratively optimize parameters through backpropagation and gradient descent.

Rather than relying on existing frameworks, every major component—from configuration and memory management to training and inference—was implemented manually in Java.

---

**Author:** Ishan Mysore
