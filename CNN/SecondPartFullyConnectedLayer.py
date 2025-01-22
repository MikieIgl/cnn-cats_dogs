import random
import numpy as np
import matplotlib.pyplot as plt


def relu(t):
    return np.maximum(0, t)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def relu_deriv(t):
    return (t >= 0).astype(float)


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc


INP_DIM = 4
OUT_DIM = 2
H_DIM = 100

ALPHA = 0.001  # скорость обучения
NUM_EPOCHS = 1000
loss_arr = []

dataset = [(np.random.randn(INP_DIM)[None, ...], random.randint(0, OUT_DIM - 1)) for _ in range(100)]

W1 = np.random.randn(INP_DIM, H_DIM)
b1 = np.random.randn(H_DIM)
W2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(OUT_DIM)

for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Forward
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax(t2)
        E = sparse_cross_entropy(z, y)

        # Backward
        y_full = to_full(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        # Update
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)

accuracy = calc_accuracy()
print(f"Accuracy: {accuracy}")

plt.plot(loss_arr)
plt.show()
