#for check work of Micrograd

from micrograd.engine import Value

#original of readme
'''
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
'''

'''
x = Value(2.0)
y = Value(4.0)
z = x*y  
print('z.grad:', z.grad)
z.backward()
print('x.grad:', x.grad)
print('y.grad:', y.grad)
'''

####check
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