import numpy as np
from criterion import Criterion


class MSE(Criterion):
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.output = np.sum(np.power(input - target, 2)) / batch_size
        return self.output
 
    def backward(self, input, target):
        self.grad_output  = (input - target) * 2 / input.shape[0]
        return self.grad_output