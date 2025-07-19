import numpy as np

# --- 1. Define the Vocabulary and Helper Functions ---

# Our "dictionary" of known words. In a real scenario, this would have thousands of words.
vocabulary = [
    "action", "account", "buy", "free", "hello",
    "meeting", "money", "now", "urgent", "verify",
    "winner", "your"
]


# A helper function to turn a sentence into a numerical vector.
# The vector will have a '1' if a word from our vocabulary exists, and '0' otherwise.
def sentence_to_vector(sentence, vocab):
    words = sentence.lower().split()
    vector = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        if word in words:
            vector[i] = 1
    return vector


# --- 2. Setup the Training Data ---
# We create a few sample "emails" and their labels (1 for Spam, 0 for Not Spam).

# Training sentences
sentences = [
    "hello, let's schedule a meeting",  # Not Spam
    "urgent action required, verify your account",  # Spam
    "buy now, free money for every winner",  # Spam
    "hello, meeting about your account"  # Not Spam
]

# The correct labels for each sentence
labels = np.array([
    [0],  # Not Spam
    [1],  # Spam
    [1],  # Spam
    [0]  # Not Spam
])

# Convert our sentences into numerical vectors for the network
# This becomes our input data, X.
X = np.array([sentence_to_vector(s, vocabulary) for s in sentences])

# For clarity, let's rename our labels to y
y = labels


# --- 3. Define the Network Architecture and Initialize ---

# Activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Set a seed for random numbers to make the results predictable
np.random.seed(42)

# Initialize weights with random values.
# The shape is determined by the connections between layers.
# Input Layer (12 neurons, one for each word) -> Hidden Layer (5 neurons)
weights0 = 2 * np.random.random((len(vocabulary), 5)) - 1
# Hidden Layer (5 neurons) -> Output Layer (1 neuron: Spam or Not Spam)
weights1 = 2 * np.random.random((5, 1)) - 1

# --- 4. The Training Loop ---
# This is where the network learns by adjusting weights and biases.
print("Training the network...")
# 20000
for j in range(20000):

    # --- FORWARD PASS ---
    # Pass the input data through the layers
    layer0 = X
    # This calculates the activations of the hidden layer neurons.
    # These neurons will learn to become our "concept detectors" (e.g., "urgency detector")
    layer1 = sigmoid(np.dot(layer0, weights0))
    # This calculates the final output of the network (the spam probability)
    layer2 = sigmoid(np.dot(layer1, weights1))

    # --- BACKWARD PASS (Backpropagation) ---
    # Calculate how wrong the network was
    layer2_error = y - layer2

    # If we are on a reporting step, print the average error
    if (j % 5000) == 0:
        print(f"Error after {j} iterations: {np.mean(np.abs(layer2_error))}")

    # Calculate the gradient for the output layer
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    # Propagate the error back to the hidden layer
    # This is the "committee meeting" where the hidden layer neurons get their "blame score"
    layer1_error = layer2_delta.dot(weights1.T)

    # Calculate the gradient for the hidden layer
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # --- UPDATE WEIGHTS (Gradient Descent) ---
    # Adjust the weights to reduce the error
    weights1 += layer1.T.dot(layer2_delta)
    weights0 += layer0.T.dot(layer1_delta)

print("Training finished.\n")


# --- 5. Test the Trained Network ---

def predict(sentence):
    print(f"Predicting for sentence: '{sentence}'")
    # Convert the sentence to a vector
    vector = sentence_to_vector(sentence, vocabulary)
    # Perform a forward pass to get the prediction
    l1 = sigmoid(np.dot(vector, weights0))
    l2 = sigmoid(np.dot(l1, weights1))

    print(f"Spam probability: {l2[0]:.2f}")
    if l2[0] > 0.5:
        print("Result: This is SPAM.\n")
    else:
        print("Result: This is NOT SPAM.\n")


# Test with a new, unseen spam email
predict("urgent! claim your free money now")

# Test with a new, unseen legitimate email
predict("hello, let's have a meeting about the account")
