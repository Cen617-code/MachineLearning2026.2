import numpy as np

def linear_layer(X, W, B):
    return (np.dot(X, W) + B)

def relu(Z):
    return np.maximum(0, Z)

X = np.array([[1.0, -2.0, 3.0]])
W1 = np.ones((3, 2))
B1 = np.array([[-1.0, -1.0]])
Z = linear_layer(X, W1, B1)
A = relu(Z)
print(A)

def forward_network(X, W1, B1, W2, B2):
    Z1 = linear_layer(X, W1, B1)
    A1 = relu(Z1)

    Z2 = linear_layer(A1 , W2, B2)
    A2 = relu(Z2)
    return A2

W2 = np.array([[-1.0],[2.0]])
B2 = np.ones((1, 1))
A2 = forward_network(X, W1, B1, W2, B2)
print(A2)

y_true = np.array([[5.0]])
lr = 0.05

for i in range(20):
    Z1 = linear_layer(X, W1, B1)
    A1 = relu(Z1)
    Z2 = linear_layer(A1, W2, B2)
    A2 = Z2

    loss = np.mean((A2 - y_true)**2)
    print(f"Iter {i}, Loss:{loss}")

    dA2 = 2 * (A2 - y_true) / X.shape[0]
    dZ2 = dA2
    dW2 = np.dot(A1.T, dZ2)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    
    dW1 = np.dot(X.T, dZ1)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= lr * dW2
    B2 -= lr * dB2
    W1 -= lr * dW1
    B1 -= lr * dB1