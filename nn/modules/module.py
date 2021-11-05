from abc import ABC, abstractclassmethod
import numpy as np


class Module(ABC):
    def __init__(self):
        self._train = True
    
    @abstractclassmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractclassmethod
    def backward(self,input, grad_output):
        raise NotImplementedError
    
    def parameters(self):
        'Возвращает список собственных параметров.'
        return []
    
    def grad_parameters(self):
        'Возвращает список тензоров-градиентов для своих параметров.'
        return []
    
    def train(self):
        self._train = True
    
    def eval(self):
        self._train = False