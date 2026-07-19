# Highly-Simplified-Scalar-Pytorch
A custom-built backpropagation engine and neural network framework using a custom Value object, featuring Graphviz visualizations. Inspiration From Andrej Karpathy Micrograd

# Project Motivation : 
While frameworks like PyTorch and TensorFlow abstract away the underlying calculus, this project was built to understand the exact mechanics of deep learning at the lowest possible level. By building the gradient tracking and chain-rule logic from scratch using pure Python, this engine demonstrates how complex neural networks learn via gradient descent without relying on black-box libraries.

This project implements backpropogation over a cutsom built DAG and includes minimal neural network consisting of Neurons , Layers , MLPs ( multi-layer perceptrons ) 

# Core Architecture 
## 1. Value Class 
The backbone of this project is a `Value` Class . Every scalar number is wrapped in this 'Value' Object to give : 
* **`data`**: The numerical float value of input scalar .
* **`grad`**: The derivative of this specific value with respect to the final value ( final value grad is 1 but for other its starts at 0.0 and adds on which each backpropogation ) . 
*   **`_backward`**: A specific function that stores how to apply the chain rule for the specific mathematical operation that created this node (e.g., addition, multiplication, ReLU).
*   **`_prev`**: A set of the child `Value` objects that produced this node (helps build the graph).
  ##  2. Topological Sort & Backpropagation
When `.backward()` is called on the final output node (usually the Loss), the engine performs a **topological sort** of the entire computational graph. It then iterates backward through the sorted nodes, recursively applying the stored `_backward` functions to distribute the gradients using the chain rule all the way to the input parameters.


