import numpy as np
import pandas as pd
import functions
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layers, learning_rate, mini_batch_size, activation, loss):
        self.layers = layers
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.num_layers = len(layers)
        self.activation_fn = {}
        self.activation_gradient = {}
        
        for i, a in enumerate(activation):
            if a == 'sigmoid':
                self.activation_fn[i + 1] = functions.sigmoid
                self.activation_gradient[i + 1] = functions.sigmoid_gradient
            elif a == 'relu':
                self.activation_fn[i + 1] = functions.relu
                self.activation_gradient[i + 1] = functions.relu_gradient
            elif a == 'softmax':
                self.activation_fn[i + 1] = functions.softmax
                self.activation_gradient[i + 1] = functions.softmax_gradient
        
        if loss == 'cross_entropy':
            self.loss = functions.cross_entropy
            self.loss_gradient = functions.cross_entropy_gradient
        elif loss == 'mse':
            self.loss = functions.mse
            self.loss_gradient = functions.mse_gradient


        self.weights = [np.array([0])] + [np.random.randn(y, x) / np.sqrt(x) for y, x in zip(layers[1:], layers[:-1])]
        self.biases = [np.array([0])] + [np.random.randn(y, 1) for y in layers[1:]]
        
        self._zs = [np.zeros((bias.shape)) for bias in self.biases] # z = w * a + b
        self._activations = [np.zeros((bias.shape)) for bias in self.biases] # a = f(z)

    def forward(self, x):
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = self.weights[i] @ self._activations[i - 1] + self.biases[i]

            self._activations[i] = self.activation_fn[i](self._zs[i])

    def backward(self, y):
        nabla_b = [np.zeros((bias.shape)) for bias in self.biases]
        nabla_w = [np.zeros((weight.shape)) for weight in self.weights]

        delta = self.loss_gradient(y, self._activations[-1]) * self.activation_gradient[self.num_layers - 1](self._zs[-1])
        
        if delta.shape != y.shape:
            delta = delta[np.argmax(y)].reshape(-1, 1)

        nabla_b[-1] = delta
        nabla_w[-1] = delta @ self._activations[-2].T

        if delta.shape != y.shape:
            delta = delta[np.argmax(y)].reshape(-1, 1)

        for i in range(self.num_layers - 2, 0, -1):
            delta = self.weights[i + 1].T @ delta * self.activation_gradient[i](self._zs[i])
            nabla_b[i] = delta
            nabla_w[i] = delta @ self._activations[i - 1].T

        return nabla_b, nabla_w
    
    def train(self, training_data, training_label, validation_data, validation_label, epochs = 10):
        loss_history = []
        for epoch in range(epochs):
            mini_batches = [
                list(zip(training_data[k: k + self.mini_batch_size], training_label[k: k + self.mini_batch_size])) for k in range(0, len(training_data), self.mini_batch_size)
            ]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros((bias.shape)) for bias in self.biases]
                nabla_w = [np.zeros((weight.shape)) for weight in self.weights]

                for x, y in mini_batch:
                    self.forward(x)
                    delta_nabla_b, delta_nabla_w = self.backward(y)

                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [w - (self.learning_rate / self.mini_batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (self.learning_rate / self.mini_batch_size) * nb for b, nb in zip(self.biases, nabla_b)]
            
            pre = self.predict_group(validation_data).reshape(validation_label.shape)
            loss = self.loss(validation_label, pre) / len(validation_data)
            loss_history.append(loss)
            accuracy = [np.argmax(pre) == np.argmax(y) for pre, y in zip(pre, validation_label)].count(True) / len(validation_data)
            print(f'Epoch {epoch}: accuracy: {accuracy:.2f} loss = {loss:.5f}')
        return loss_history

    def evaluate(self, test_data, test_label):
        results = [np.argmax(self.predict(x)) == np.argmax(y) for x, y in zip(test_data, test_label)]
        return sum(results)

    def predict_group(self, data):
        res = np.array([])
        for x in data:
            res = np.append(res, self.predict(x))
        return res
    
    def predict(self, x):
        self.forward(x)
        return self._activations[-1]
    
    def save(self, path = 'model.npy'):
        np.save(path, [self.weights, self.biases])
    
    def load(self, path = 'model.npy'):
        self.weights, self.biases = np.load(path, allow_pickle=True)
        
def load_mnist():
    train_data = pd.read_csv('../dataset/mnist/mnist_train.csv').values
    test_data = pd.read_csv('../dataset/mnist/mnist_test.csv').values

    train_images = train_data[:, 1:]
    train_images = train_images / 255.0
    train_labels_tmp = train_data[:, :1]
    train_images = np.array([i.reshape(-1, 1) for i in train_images])
    train_labels = np.zeros((len(train_labels_tmp), 10))
    for i in range(len(train_labels_tmp)):
        train_labels[i][train_labels_tmp[i]] = 1
    train_labels = np.array([i.reshape(-1, 1) for i in train_labels])

    test_images = test_data[:, 1:]
    test_images = test_images / 255.0
    test_labels_tmp = test_data[:, :1]
    test_images = np.array([i.reshape(-1, 1) for i in test_images])
    test_labels = np.zeros((len(test_labels_tmp), 10))
    for i in range(len(test_labels_tmp)):
        test_labels[i][test_labels_tmp[i]] = 1
    test_labels = np.array([i.reshape(-1, 1) for i in test_labels])

    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist()
    net = NeuralNetwork([784, 25, 10], 0.01, 16, ['relu', 'softmax'], loss = 'cross_entropy')
    cost_history = net.train(X_train, y_train, X_test, y_test, 10)
    # net.load('model.npy')
    # plt.plot(cost_history)
    # plt.show()
