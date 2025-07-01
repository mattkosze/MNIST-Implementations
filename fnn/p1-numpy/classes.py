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

    # The backward method which calculate the partial derivatives
    def backward(self, dvalues):
        # Let's copy the dvalues so we can modify them directly
        self.dinputs = dvalues.copy()

        # Give a zero gradient where values were negative - therefore not fulfilling the (x>0)
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    # Method for the forward pass; no init overriding needed
    def forward(self, inputs):
        # Calculate the non-normalized probabilities using np.exp(), which exponentiates everything, np.max() which finds the maximum value in the matrix, and np.sum() which performs a summation of each row/col as desired
        unProb = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize the inputs for every sample
        nProb = unProb / np.sum(unProb, axis=1, keepdims=True)

        self.output = nProb

    # Our backward pass method
    def backward(self, dvalues):
        # Create an () empty array, using the np.empty_like() method.
        self.dinputs = np.empty_like(dvalues)

        for index, (singleOutput, singleDValues) in enumerate(zip(self.output, dvalues)):
            # Flatten our output array
            singleOutput = singleOutput.reshape(-1, 1)
            # Calculate our Jacobian matrix using the technique showed above
            jacobian = (np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T))

            # Calculate our sample-wise gradient and add it to the correct index
            self.dinputs[index] = np.dot(jacobian, singleDValues)


class CategoricalCrossEntropyLoss:
    # The forward method because we don't need to override any initializations
    def forward(self, yPred, yTrue):
        # Number of samples in the batch
        samples = len(yPred)
        # Clip our data from both ends to prevent division by 0
        yPred = np.clip(yPred, 1e-7, 1 - 1e-7)

        # Probabilities for one-hot encoded target values
        confidences = np.sum(yPred * yTrue, axis=1)
        # Losses
        negLogLikelihood = -np.log(confidences)

        return negLogLikelihood

    # Our backward pass method; we assume all labels are one-hot
    def backward(self, dValues, yTrue):
        # Figure out the amount of samples
        sampleLen = len(dValues)
        # Calculate our gradient and normalize it
        self.dinputs = (-yTrue / dValues) / sampleLen

    # A calculate method
    def calculate(self, output, y):
        # Calculate the loss
        sampleLoss = self.forward(output, y)
        # Calculate the mean loss
        dataLoss = np.mean(sampleLoss)

        return dataLoss