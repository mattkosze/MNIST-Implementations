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