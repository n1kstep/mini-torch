from module import Module


class Sequential(Module):
    def __init__ (self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        """
        Прогоните данные последовательно по всем слоям:
        
            y[0] = layers[0].forward(input)
            y[1] = layers[1].forward(y_0)
            ...
            output = module[n-1].forward(y[n-2])   
            
        Это должен быть просто небольшой цикл: for layer in layers...
        
        Хранить выводы ещё раз не надо: они сохраняются внутри слоев после forward.
        """

        for layer in self.layers:
            input = layer.forward(input)

        self.output = input
        return self.output

    def backward(self, input, grad_output):
        """
        Backward -- это как forward, только наоборот. (с)
        
        Предназначение backward:
        1. посчитать посчитать градиенты для собственных параметров
        2. передать градиент относительно своего входа
        
        О своих параметрах модули сами позаботятся. Нам же нужно позаботиться о передачи градиента.
         
            g[n-1] = layers[n-1].backward(y[n-2], grad_output)
            g[n-2] = layers[n-2].backward(y[n-3], g[n-1])
            ...
            g[1] = layers[1].backward(y[0], g[2])   
            grad_input = layers[0].backward(input, g[1])
        
        Тут цикл будет уже чуть посложнее.
        """
        
        for i in range(len(self.layers)-1, 0, -1):
            grad_output = self.layers[i].backward(self.layers[i-1].output, grad_output)
        
        grad_input = self.layers[0].backward(input, grad_output)
        
        return grad_input
      
    def parameters(self):
        'Можно просто сконкатенировать все параметры в один список.'
        res = []
        for l in self.layers:
            res += l.parameters()
        return res
    
    def grad_parameters(self):
        'Можно просто сконкатенировать все градиенты в один список.'
        res = []
        for l in self.layers:
            res += l.grad_parameters()
        return res
    
    def train(self):
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        for layer in self.layers:
            layer.eval()