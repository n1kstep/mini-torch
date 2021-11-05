import numpy as np
from module import Module


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super().__init__()
            
        self.slope = slope
        
    def forward(self, input):
        self.output = np.maximum(input, input * self.slope)
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0) + np.multiply(grad_output, (input <= 0) * self.slope)
        return grad_input