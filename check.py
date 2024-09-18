#for check work of Micrograd

from micrograd.engine import Value

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.datasets import make_moons, make_blobs

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
        print("-----")
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

class MSELoss():
    def forward(self, y_true, y_pred):
        #todo: is istance and isnt istance
        y_true = np.array(y_true.data)
        y_pred = np.array(y_pred.data)
        output = np.mean((y_true-y_pred)**2)
        return output 
    
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
    
    def __repr__(self):
        return("MSELoss()")

def loss_example():
    x, y = make_regression(n_samples=10, n_features=1, noise=10, random_state=0)
    x = np.interp(x, (x.min(), x.max()), (10, 20))
    y = np.interp(y, (y.min(), y.max()), (5, 15))
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)   
    print(xTrain)
    print(yTrain) 
    linear_layer = Linear(1, 1)
    outputs_ = linear_layer(xTrain)
    #print("type(outputs_):",type(outputs_)) #is Value
    yTrain_ = yTrain.reshape(7,1)
    #print(type(yTrain_)) #np.array
    criterion = MSELoss()
    loss = criterion(Value(yTrain_), outputs_) #Value, Value
    #loss = criterion(yTrain_, outputs_) #np.array, np.array
    print("MESLoss:")
    print(loss)

#loss_example()

from sklearn.datasets import make_moons, make_blobs

def demo_example():
    # make up a dataset
    X, y = make_moons(n_samples=10, noise=0.1)
    print(y)
    y = y*2 - 1 # make y be -1 or 1
    print(X)
    print(y)
    
#demo_example()

def demo_pytorch():
    # Generar el dataset
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1  # Convertir a -1 o 1

    # Convertir a tensores de PyTorch
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)  # Cambiar la forma para que sea 2D

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Definir el modelo
    class MLP(nn.Module):
        def __init__(self, nin, nouts):
            super(MLP, self).__init__()
            sz = [nin] + nouts
            self.layers = nn.ModuleList([nn.Linear(sz[i], sz[i + 1]) for i in range(len(nouts))])

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i != len(self.layers) - 1:  # Aplicar ReLU solo en capas ocultas
                    x = torch.relu(x)
            return x

    model = MLP(2, [16, 16, 1])  # Red neuronal de 2 capas ocultas
    print(model)
    print("Número de parámetros:", sum(p.numel() for p in model.parameters()))

    # Función de pérdida
    def loss(batch_size=None):
        if batch_size is None:
            Xb, yb = X_train, y_train
        else:
            ri = np.random.permutation(X_train.shape[0])[:batch_size]
            Xb, yb = X_train[ri], y_train[ri]

        # Forward
        scores = model(Xb)

        # Pérdida SVM "max-margin"
        losses = (1 + -yb * scores).clamp(min=0)  # ReLU
        data_loss = losses.mean()

        # Regularización L2
        alpha = 1e-4
        reg_loss = alpha * sum((p**2).sum() for p in model.parameters())

        total_loss = data_loss + reg_loss

        # Precisión
        accuracy = ((yb > 0) == (scores > 0)).float().mean()
        return total_loss, accuracy

    # Entrenamiento
    for k in range(100):
        # Forward
        total_loss, acc = loss()

        # Backward
        model.zero_grad()
        total_loss.backward()

        # Actualización (SGD)
        learning_rate = 1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 1 == 0:
            print(f"Paso {k} - Pérdida {total_loss.item():.4f}, Precisión {acc.item() * 100:.2f}%")

    # Visualizar la frontera de decisión
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])  # Convertir a tensor de PyTorch

    # Hacer predicciones
    with torch.no_grad():
        scores = model(Xmesh)
        Z = (scores > 0).numpy()  # Convertir a numpy y aplicar la clasificación

    Z = Z.reshape(xx.shape)

    # Crear el gráfico
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

demo_pytorch()

