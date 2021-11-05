import numpy as np
from module import Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = 1 / (1 + np.exp())
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.dot(self.forward(input), 1 - self.forward(input))
        return grad_input