import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
if __name__ == "__main__":
    # Training Data (AND logic gate)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(learning_rate=0.1, n_iter=10)
    perceptron.fit(X, y)

    # Predictions 
    print(perceptron.predict(X))