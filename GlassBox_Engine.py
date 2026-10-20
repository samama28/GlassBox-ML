
import numpy as np


class GlassBoxMLP:
    def __init__(
        self,
        input_dim=2,
        hidden_dim=8,
        output_dim=1,
        learning_rate=0.5
    ):
        self.lr = learning_rate

        # He initialization for the ReLU hidden layer
        self.W1 = (
            np.random.randn(input_dim, hidden_dim)
            * np.sqrt(2.0 / input_dim)
        )
        self.b1 = np.zeros((1, hidden_dim))

        # Xavier initialization for the sigmoid output layer
        self.W2 = (
            np.random.randn(hidden_dim, output_dim)
            * np.sqrt(1.0 / hidden_dim)
        )
        self.b2 = np.zeros((1, output_dim))

    # Activation functions
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))

    # Forward propagation
    def forward(self, X):
        self.X = X

        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    # Binary cross-entropy loss
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-15

        y_pred = np.clip(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        loss = -(1 / m) * np.sum(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return loss

    # Backpropagation and parameter updates
    def backward(self, y_true):
        m = y_true.shape[0]

        # Output layer gradients
        dZ2 = self.A2 - y_true

        dW2 = (1 / m) * np.dot(
            self.A1.T,
            dZ2
        )
        db2 = (1 / m) * np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)

        dW1 = (1 / m) * np.dot(
            self.X.T,
            dZ1
        )
        db1 = (1 / m) * np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )

        # Update weights and biases using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # Train the model for one epoch
    def train_epoch(self, X, y):
        y_pred = self.forward(X)

        loss = self.compute_loss(y, y_pred)

        self.backward(y)

        accuracy = np.mean(
            (y_pred >= 0.5) == y
        )

        return loss, accuracy


if __name__ == "__main__":

    # XOR dataset: a non-linearly separable pattern
    X_xor = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ],
        dtype=float
    )

    y_xor = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ],
        dtype=float
    )

    print("Starting GlassBox Engine Training on XOR Pattern...\n")

    # Initialize the neural network
    model = GlassBoxMLP(
        input_dim=2,
        hidden_dim=8,
        output_dim=1,
        learning_rate=0.5
    )

    # Train the model for 2,000 epochs
    for epoch in range(1, 2001):
        loss, accuracy = model.train_epoch(
            X_xor,
            y_xor
        )

        if epoch % 400 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss:.5f} | "
                f"Accuracy: {accuracy * 100:.1f}%"
            )

    # Display final predictions
    print("\n--- Final Predictions ---")

    predictions = model.forward(X_xor)

    for x_i, target, pred in zip(
        X_xor,
        y_xor,
        predictions
    ):
        predicted_class = int(pred[0] >= 0.5)

        print(
            f"Input: {x_i} -> "
            f"Target: {int(target[0])} | "
            f"Predicted Probability: {pred[0]:.4f} | "
            f"Class: {predicted_class}"
        )
