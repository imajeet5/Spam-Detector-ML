import numpy as np


# --- 1. Define the Activation Function and its Derivative ---
# We use the sigmoid function, which squishes numbers into a 0-1 range.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid is needed for backpropagation (the chain rule).
def sigmoid_derivative(x):
    return x * (1 - x)


# --- 2. Setup the Network and Data ---
# Input data (e.g., a tiny 4-pixel black-and-white image)
# Each row is a different training example.
X = np.array([[0, 0, 1, 1],
              [0, 1, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 1, 0]])

# Output data (the correct answers for each input)
y = np.array([[0],
              [1],
              [1],
              [0]])

# --- 3. Initialize Weights and Biases Randomly ---
# This is the starting point. The network knows nothing.
# We use np.random.seed for reproducibility, so we get the same random numbers each time.
np.random.seed(1)

# Weights connecting the input layer (4 neurons) to the hidden layer (5 neurons)
weights0 = 2 * np.random.random((4, 5)) - 1
# Weights connecting the hidden layer (5 neurons) to the output layer (1 neuron)
weights1 = 2 * np.random.random((5, 1)) - 1

# Note: For simplicity, this example omits biases. In a real network, they would be here.

# --- 4. The Training Loop ---
# We will repeat this process many times to "learn".
for j in range(200000):
    # --- FORWARD PASS: Feed the data through the network ---
    # Layer 0 is the input data
    layer0 = X

    # Calculate the activations for the hidden layer (layer 1)
    # This is the (weighted sum) part: np.dot(layer0, weights0)
    # Then we apply the sigmoid function.
    layer1 = sigmoid(np.dot(layer0, weights0))

    # Calculate the activations for the output layer (layer 2)
    layer2 = sigmoid(np.dot(layer1, weights1))

    # --- BACKWARD PASS: Calculate the error and gradients ---

    # Step 1: Calculate the error for the final layer.
    # This is how far off the prediction (layer2) was from the true answer (y).
    layer2_error = y - layer2

    # Step 2: Calculate the "slope" or gradient for the final layer.
    # We multiply the error by the derivative of the sigmoid at layer2's activation values.
    # This tells us how much to change the weights to reduce the error.
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    # Step 3: Propagate the error back to the hidden layer.
    # How much did layer 1 contribute to the layer 2 error?
    # We use the weights (weights1) to distribute the blame.
    layer1_error = layer2_delta.dot(weights1.T)

    # Step 4: Calculate the gradient for the hidden layer.
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # --- UPDATE WEIGHTS: Take a small step downhill ---

    # Update the weights for the connection between the hidden layer and output layer
    # We adjust them based on the activations of the hidden layer (layer1) and the output gradient (layer2_delta).
    weights1 += layer1.T.dot(layer2_delta)

    # Update the weights for the connection between the input layer and hidden layer
    weights0 += layer0.T.dot(layer1_delta)

# --- After training is finished ---
print("Output After Training:")
print(layer2)
print("\nExpected Output:")
print(y)
