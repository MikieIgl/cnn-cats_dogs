import numpy as np


def relu(t):
    return np.maximum(0, t)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


inp_dim = 3
out_dim = 2
h_dim = 3

x = np.random.randn(inp_dim)

W1 = np.random.randn(inp_dim, h_dim)
b1 = np.random.randn(h_dim)
W2 = np.random.randn(h_dim, out_dim)
b2 = np.random.randn(out_dim)

probs = predict(x)
pred_class = np.argmax(probs)
class_names = ['Cat', 'Dog']
print(f'Predicted class: {class_names[pred_class]}')
print(probs, sum(probs))
