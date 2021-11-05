import numpy as np
from abc import ABCMeta, abstractmethod


class Criterion(metaclass=ABCMeta):   

    @abstractmethod     
    def forward(self, input, target):
        raise NotImplementedError

    @abstractmethod
    def backward(self, input, target):
        raise NotImplementedError