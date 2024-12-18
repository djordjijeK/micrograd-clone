import random

from abc import ABC, abstractmethod
from micrograd_clone.engine import Value


class Module(ABC):

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0

    @abstractmethod
    def parameters(self):
        raise NotImplemented


class Neuron(Module):

    def __init__(self, n_inputs, nonlinearity = True):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.__nonlinearity = nonlinearity


    def __call__(self, vec, *args, **kwargs):
        activation = sum((w*x for w,x in zip(self.weights, vec)), self.bias)
        return activation.relu() if self.__nonlinearity else activation


    def parameters(self):
      return self.weights + [self.bias]


    def __repr__(self):
        return f"{'ReLU' if self.__nonlinearity else 'Linear'}Neuron({len(self.weights)})"


class Layer(Module):

    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]


    def __call__(self, x, *args, **kwargs):
        output = [neuron(x) for neuron in self.neurons]
        return output[0] if len(self.neurons) == 1 else output

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]


    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MultilayerPerceptron(Module):

    def __init__(self, n_inputs, layers):
        layout = [n_inputs] + layers
        self.layers = [Layer(layout[i], layout[i + 1], nonlinearity= i!=len(layers)-1) for i in range(len(layers))]


    def __call__(self, vec, *args, **kwargs):
        for layer in self.layers:
            vec = layer(vec)

        return vec


    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]


    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"