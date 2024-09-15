#for check work of Micrograd

from micrograd.engine import Value

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np 

def readme_example():
    #original of readme
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
    g.backward()
    print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
    print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

def simple_example():
    x = Value(2.0)
    y = Value(4.0)
    z = x*y  
    print('z.grad:', z.grad)
    z.backward()
    print('x.grad:', x.grad)
    print('y.grad:', y.grad)

def check_example():
    # inputs x1,x2
    x1 = Value(2.0)
    x2 = Value(0.0)
    # weights w1,w2
    w1 = Value(-3.0)
    w2 = Value(1.0)
    # bias of the neuron
    b = Value(6.8813735870195432)
    # x1*w1 + x2*w2 + b
    x1w1 = x1*w1 
    x2w2 = x2*w2 
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    # ----
    e = (2*n).exp()
    o = (e - 1) / (e + 1)
    # ----
    print(o.backward())
    print('o.grad:', o.grad)
    print('e.grad:', e.grad)
    print('n.grad:', n.grad)
    print('x1w1.grad:', x1w1.grad)
    print('x1.grad:', x1.grad)



class Linear:
    def __init__(self, in_features, out_features, bias=True):
        """
        Inicialización de weights con Xavier/Glorot
        """
        self.in_features = in_features
        self.out_features = out_features
        self.status_bias = bias
        limit = np.sqrt(6 / (in_features + out_features))
        self.weights = Value(np.random.uniform(-limit, limit, (out_features, in_features)))
        print("weights:")
        print(self.weights)
        self.bias = Value(np.zeros(out_features))
        print("bias:")
        print(self.bias)

    def forward(self, x):
        return Value(x.data @ self.weights.data.T + self.bias.data)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.status_bias})"

def linear_regression():
    x, y = make_regression(n_samples=10, n_features=1, noise=10, random_state=0)
    x = np.interp(x, (x.min(), x.max()), (10, 20))
    y = np.interp(y, (y.min(), y.max()), (5, 15))
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)   
    print(xTrain)
    print(yTrain) 
    inputs = Value(xTrain)
    print(inputs)
    linear_layer = Linear(1, 1)
    print(linear_layer)
    outputs = linear_layer(inputs)
    print("outputs:")
    print(outputs)





linear_regression()
    











