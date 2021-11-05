from abc import ABCMeta, abstractmethod
import numpy as np


class Module(ABCmetaclass=ABCMeta):
    def __init__(self):
        self._train = True
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, input, grad_output):
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