import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def plot_sigmoid(start, end, dots):
    # Генерация данных
    X = np.linspace(start, end, dots)
    Y = sigmoid(X)

    # Создание графика
    plt.plot(X, Y, color='blue')
    plt.xlabel('X')
    plt.ylabel('Sigmoid(X)')
    plt.title('Сигмоидальная функция')

    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.axhline(1, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    plt.xticks(np.arange(start, end + 1, 1))
    plt.yticks(np.arange(round(min(Y), 1), round(max(Y), 1) + 0.1, 0.1))

    plt.grid()
    plt.show()

def plot_tanh(start, end, dots):
    # Генерация данных
    X = np.linspace(start, end, dots)
    Y = tanh(X)

    # Создание графика
    plt.plot(X, Y, color='blue')
    plt.xlabel('X')
    plt.ylabel('Tanh(X)')
    plt.title('Гиперболический тангенс')

    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.axhline(1, color='black', linewidth=1, linestyle='--')
    plt.axhline(-1, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    plt.xticks(np.arange(start, end + 1, 1))
    plt.yticks(np.arange(round(min(Y), 1), round(max(Y), 1) + 0.1, 0.1))

    plt.grid()
    plt.show()


def plot_relu(start, end, dots):
    # Генерация данных
    X = np.linspace(start, end, dots)
    Y = [relu(dot) for dot in X]

    # Создание графика
    plt.plot(X, Y, color='blue')
    plt.xlabel('X')
    plt.ylabel('ReLU(X)')
    plt.title('Линейная ректификация')

    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    plt.xticks(np.arange(start, end + 1, 1))
    plt.yticks(np.arange(min(Y), max(Y) + 1, 1))

    plt.grid()
    plt.show()


plot_sigmoid(-10, 10, 1000)
plot_tanh(-10, 10, 1000)
plot_relu(-10, 10, 100)
