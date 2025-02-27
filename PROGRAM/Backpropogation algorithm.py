import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, Y):
        # Backward pass (gradient descent)
        m = X.shape[0]  # number of samples
        dZ2 = self.A2 - Y  # error at output layer
        dW2 = np.dot(self.A1.T, dZ2) / m  # gradient for weights in the second layer
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # gradient for bias in second layer
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)  # error at hidden layer
        dW1 = np.dot(X.T, dZ1) / m  # gradient for weights in the first layer
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # gradient for bias in first layer
        
        # Update weights and biases using the gradients
        self.W1 -= dW1
        self.b1 -= db1
        self.W2 -= dW2
        self.b2 -= db2

    def train(self, X, Y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            self.forward(X)  # Forward pass
            self.backward(X, Y)  # Backward pass
            
            if epoch % 100 == 0:  # Print loss every 100 epochs
                loss = np.mean((self.A2 - Y) ** 2)  # Mean squared error loss
                print(f"Epoch {epoch} - Loss: {loss}")

# Example usage:
if __name__ == "__main__":
    # Sample data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
    Y = np.array([[0], [1], [1], [0]])  # Target output

    # Initialize and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)  # 2 input features, 4 hidden neurons, 1 output neuron
    nn.train(X, Y, epochs=1000, learning_rate=0.1)

    # Test after training
    predictions = nn.forward(X)
    print("\nPredictions after training:")
    print(predictions)
