import numpy as np

class MLPFromScratch:
    def __init__(self, layer_dims):
        """
        layer_dims: List containing the number of neurons in each layer.
        Example: [input_features, hidden_1, hidden_2, output_neurons]
        """
        self.layer_dims = layer_dims
        self.params = self._initialize_parameters()
        
    def _initialize_parameters(self):
        """He Initialization: prevents gradients from exploding/vanishing."""
        np.random.seed(42)
        params = {}
        for l in range(1, len(self.layer_dims)):
            # Weight shape: (current_layer, previous_layer)
            params[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2/self.layer_dims[l-1])
            params[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return params

    # --- Activation Functions ---
    def relu(self, Z): return np.maximum(0, Z)
    
    def sigmoid(self, Z): return 1 / (1 + np.exp(-Z))

    def relu_derivative(self, Z): return Z > 0

    # --- Forward & Backward Core ---
    def forward(self, X):
        """Transmits input through the network."""
        cache = {"A0": X}
        A = X
        L = len(self.layer_dims) - 1

        for l in range(1, L + 1):
            Z = np.dot(self.params[f'W{l}'], A) + self.params[f'b{l}']
            # Use Sigmoid for final layer, ReLU for hidden layers
            A = self.sigmoid(Z) if l == L else self.relu(Z)
            
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        return A, cache

    def backward(self, Y, cache):
        """Calculates gradients using the Chain Rule."""
        m = Y.shape[1]
        grads = {}
        L = len(self.layer_dims) - 1
        
        # 1. Output Layer Gradient (Derivative of Cross-Entropy w.r.t Z)
        dZ = cache[f'A{L}'] - Y 
        
        for l in range(L, 0, -1):
            grads[f'dW{l}'] = (1/m) * np.dot(dZ, cache[f'A{l-1}'].T)
            grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1: # Calculate dZ for the next layer moving backwards
                dA_prev = np.dot(self.params[f'W{l}'].T, dZ)
                dZ = dA_prev * self.relu_derivative(cache[f'Z{l-1}'])
                
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, len(self.layer_dims)):
            self.params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * grads[f'db{l}']

    def train(self, X, Y, epochs=1000, lr=0.01):
        for i in range(epochs):
            A_final, cache = self.forward(X)
            grads = self.backward(Y, cache)
            self.update_parameters(grads, lr)
            
            if i % 100 == 0:
                loss = -np.mean(Y * np.log(A_final + 1e-8) + (1-Y) * np.log(1-A_final + 1e-8))
                print(f"Epoch {i} | Loss: {loss:.4f}")

# --- Example Usage with Synthetic Data ---
if __name__ == "__main__":
    # Input: 2 features (like height/weight)
    # Target: 1 binary label (0 or 1)
    X_train = np.random.randn(2, 100) 
    Y_train = (np.sum(X_train, axis=0) > 0).astype(int).reshape(1, 100)

    # Define Architecture: 2 inputs -> 4 hidden neurons -> 1 output
    model = MLPFromScratch(layer_dims=[2, 4, 1])
    model.train(X_train, Y_train, epochs=500, lr=0.1)
