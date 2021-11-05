from nn.module import Module


class Linear(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Xavier initialization: инциализируем так,
        # что если на вход идет N(0, 1)
        # то и на выходе должно идти N(0, 1)
        stdv = 1./np.sqrt(dim_in)
        self.W = np.random.uniform(-stdv, stdv, size=(dim_in, dim_out))
        self.b = np.random.uniform(-stdv, stdv, size=dim_out)
        
    def forward(self, input):
        self.output = np.dot(input, self.W) + self.b
        return self.output
    
    def backward(self, input, grad_output):
        self.grad_b = np.mean(grad_output, axis=0)
        
        #     in_dim x batch_size
        self.grad_W = np.dot(input.T, grad_output)
        #                 batch_size x out_dim
        
        grad_input = np.dot(grad_output, self.W.T)
        
        return grad_input
    
    def parameters(self):
        return [self.W, self.b]
    
    def grad_parameters(self):
        return [self.grad_W, self.grad_b]