import numpy as np

class DenseLayer:
    # Initialization method to initialize a dense layer object
    def __init__(self, numInputs, numNeurons):
        # Create random weights and scale them down, of size (inputs x neurons)
        self.weights = 0.01 * np.random.randn(numInputs, numNeurons)
        # Create an array filled with 0's with a bias for each neuron
        self.biases = np.zeros((1, numNeurons))

    # Our forward pass method
    def forward(self, inputs):
        # Calculate outputs using the method previously discussed
        self.outputs = np.dot(inputs, self.weights) + self.biases

class ReLU:
    # Our forward method; no overriding init method needed
    def forward(self, inputs):
        # Using the np.max() method to carry out our comparisons
        self.output = np.maximum(0, inputs)


class Softmax:
    # Method for the forward pass; no init overriding needed
    def forward(self, inputs):
        # Calculate the non-normalized probabilities using np.exp(), which exponentiates everything, np.max() which finds the maximum value in the matrix, and np.sum() which performs a summation of each row/col as desired
        unProb = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize the inputs for every sample
        nProb = unProb / np.sum(unProb, axis=1, keepdims=True)

        self.output = nProb