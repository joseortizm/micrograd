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
    s = z.sigmoid()
    s.backward()
    print('s:', s)
    print('s.grad:', s.grad)
    print('z.grad:', z.grad)
    print('x.grad:', x.grad)
    print('y.grad:', y.grad)


simple_example()



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

    #TODO backward
    def backward(self):
        """
        Calcula el gradiente de la pérdida con respecto a las predicciones.
        
        :return: Gradiente de la pérdida
        """
        n = self.y_true.shape[0]
        print(n)
        # Gradiente: -2 * (y_true - y_pred) / n
        return 2 * (self.y_pred - self.y_true) / n


    def __repr__(self):
        return("MSELoss()")


class _MSELoss():
    def forward(self, y_pred, y_true):
        #todo: is istance and isnt istance
        #y_true = np.array(y_true.data)
        #y_pred = np.array(y_pred.data)
        #output = np.mean((y_true-y_pred)**2)
        output = sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)
        #output = Value(output)
        return output 
    
    def __call__(self, y_true, y_pred):
        return self.forward(y_pred, y_true)

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
    loss = Value(loss)
    loss.backward()
    print(yTrain_.grad) #TODO a Value(): attributeError: 'numpy.ndarray' object has no attribute 'grad'

#loss_example()


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

#demo_pytorch()


def linearRegression_Value():
    # Generar datos de regresión
    x, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=0) 
    #n_samples > 100000: in build_topo visited.add(v) RecursionError: maximum recursion depth exceeded while calling a Python object
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

    # Normalizar datos (Min-Max)
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    xTrain_norm = normalize(xTrain)
    xTest_norm = normalize(xTest)
    yTrain_norm = normalize(yTrain)
    yTest_norm = normalize(yTest)

    # Clase para la regresión lineal
    class LinearRegression:
        def __init__(self):
            self.w = Value(np.random.randn())  # Peso
            self.b = Value(np.random.randn())  # Sesgo

        def forward(self, x):
            return x * self.w + self.b

    # Función de pérdida (MSE)
    def mse_loss(y_pred, y_true):
        return sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)

    # Clase para el optimizador SGD
    class SGD:
        def __init__(self, params, lr=0.01):
            self.params = params
            self.lr = lr

        def step(self):
            for param in self.params:
                param.data -= self.lr * param.grad
                param.grad = 0  # Resetear gradiente después de la actualización

    # Inicializar el modelo y el optimizador
    model = LinearRegression()
    optimizer = SGD([model.w, model.b], lr=0.1)
    epochs = 100 #earlystop podriamos aplicar en 15 segun grafica del error; comparaciones si modifico lr y spochs
    losses = []

    #TODO early stopping
    #patience_counter = 0
    #patience = 10
    #best_loss = float('inf')
    criterion = _MSELoss()

    # Entrenamiento
    for epoch in range(epochs):
        epoch += 1

        # Convertir datos a valores de Micrograd
        inputs = np.array([Value(float(i)) for i in xTrain_norm.flatten()])
        labels = np.array([Value(float(i)) for i in yTrain_norm.flatten()])

        # Forward pass
        predictions = np.array([model.forward(x) for x in inputs])

        # Calcular pérdida
        #loss = mse_loss(predictions, labels) #original
        loss = criterion(predictions, labels)
        losses.append(loss)

        # Backward pass
        loss.backward()

        # Actualizar parámetros
        optimizer.step()

        print(f'Epoch: {epoch} | Loss: {loss.data}')

        #TODO -> Early stopping
        #if loss.data < best_loss:
        #    best_loss = loss.data
        #    patience_counter = 0  # Reiniciar contador
        #else:
        #    patience_counter += 1
        #
        #if patience_counter >= patience:
        #    print("Early stopping activated.")
        #    break
#
    # Predicción en el conjunto de prueba
    test_inputs = np.array([Value(float(i)) for i in xTest_norm.flatten()])
    test_predictions = np.array([model.forward(x) for x in test_inputs])

    # Mostrar resultados (opcional)
    #for i in range(len(xTest)):
    #    print(f'Input: {xTest[i][0]} | Predicción: {test_predictions[i].data} | Verdadero: {yTest_norm[i]}')
    
    print("model.w.data: ",model.w.data) 
    print("model.b.data:", model.b.data) 
   
    ## Graficar resultados
    
    # Definir los parámetros de la recta
    m = model.w.data  # Pendiente
    b = model.b.data  # Intersección en y

    # Crear valores de x
    #x = np.linspace(-10, 10, 100)  # 100 puntos entre -10 y 10
    x_ = xTest_norm 

    # Calcular los valores de y según la ecuación de la recta
    y_ = m * x_ + b

    # Graficar la recta
    plt.scatter(xTest_norm, yTest_norm, color='orange', label='Datos de prueba')
    plt.plot(x_, y_, color='red', label=f'y = {m}x + {b}')
    plt.title('Graficar una Recta')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.axhline(0, color='black',linewidth=0.5, ls='--')  # Línea horizontal en y=0
    #plt.axvline(0, color='black',linewidth=0.5, ls='--')  # Línea vertical en x=0
    #plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)  # Cuadrícula
    plt.legend()
    #plt.xlim(-10, 10)  # Límites del eje x
    #plt.ylim(-10, 10)  # Límites del eje y
    plt.show()

    ##Graficar Losses
    losses_ = [l.data for l in losses]
    plt.plot(losses_, color='blue', label='Pérdida (Loss)')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

#linearRegression_Value()




#TODO:
def pytorch_loss_example():
    #https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    loss = nn.MSELoss()
    input = torch.randn(3, 5, requires_grad=True)
    print("input:")
    print(input)
    target = torch.randn(3, 5)
    print("target:")
    print(target)
    output = loss(input, target)
    print("output:")
    print(type(output))
    print(output)
    print("backward():")
    output.backward()
    print(input.grad)
    print(target.grad) #None
    np_input = input.detach().numpy()
    print(np_input)
    input_v = Value(np_input)
    print(input_v)
    target_v = Value(target.detach().numpy())
    print(target_v)
    
    criterion_v = MSELoss()
    output_v = criterion_v(input_v, target_v)
    output_v = Value(output_v)
    print(output_v) 
    output_v.backward()
    print(input_v.grad)
    


#pytorch_loss_example()
