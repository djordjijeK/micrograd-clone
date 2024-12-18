class Value:

    def __init__(self, data, parents=(), operation=''):
        self.__data = data
        self.__parents = set(parents)
        self.__operation = operation

        self.__grad = 0.0
        self._backward = lambda: None

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    @property
    def parents(self):
        return self.__parents

    @property
    def operation(self):
        return self.__operation

    @property
    def grad(self):
        return self.__grad

    @grad.setter
    def grad(self, grad):
        self.__grad = grad

    # self + other
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # gradients are accumulated for different training examples
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward

        return output

    # self * other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # gradients are accumulated for different training examples
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward

        return output

    # self ** other
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        output = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # gradients are accumulated for different training examples
            self.grad += (other * self.data ** (other - 1)) * output.grad

        output._backward = _backward

        return output

    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # gradients are accumulated for different training examples
            self.grad += (output.data > 0) * output.grad

        output._backward = _backward

        return output

    def backward(self):
        order = []
        visited = set()

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                order.append(node)

        dfs(self)

        self.grad = 1
        for node in reversed(order):
            node._backward()

    # other + self
    def __radd__(self, other):
        return self + other

    # -self
    def __neg__(self):
        return self * -1

    # other - self
    def __rsub__(self, other):
        return other + (-self)

    # self - other
    def __sub__(self, other):
        return self + (-other)

    # other * self
    def __rmul__(self, other):
        return other * self

    # self / other
    def __truediv__(self, other):
        return self * other ** -1

    # other / self
    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        parts = [f"Value(data: {self.data}, grad: {self.__grad}"]

        if self.parents:
            parents_str = ", ".join(str(parent.data) for parent in self.parents)
            parts.append(f", operation: {self.operation}({parents_str})")

        return "".join(parts) + ")"