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


# --- 2. Activation function (sigmoid) and its derivative ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# --- 3. Initialize Weights (Globally, so they persist across function calls) ---
# Set a seed for random numbers to make the results predictable
np.random.seed(42)

# Input Layer (12 neurons) -> Hidden Layer (5 neurons)
weights0 = 2 * np.random.random((len(vocabulary), 5)) - 1
# Hidden Layer (5 neurons) -> Output Layer (1 neuron)
weights1 = 2 * np.random.random((5, 1)) - 1


# --- 4. Function to Train on One Example ---
def train_on_example(sentence, label, iterations=1, learning_rate=1.0):
    """
    Trains the network on a single example for a given number of iterations.

    Args:
    - sentence (str): The input sentence (e.g., "urgent action required").
    - label (int): 0 for not spam, 1 for spam.
    - iterations (int): How many times to update on this example (default 1 for pure SGD).
    - learning_rate (float): Scaling factor for updates (default 1.0, adjust if needed).

    This function uses global weights0 and weights1, updating them in place.
    """
    global weights0, weights1  # Use the shared weights

    # Convert sentence to vector (1x12)
    vector = sentence_to_vector(sentence, vocabulary)
    layer0 = vector.reshape(1, -1)  # Ensure it's 2D: (1, 12)

    # Convert label to array (1x1)
    y = np.array([[label]])

    for _ in range(iterations):
        # --- FORWARD PASS ---
        layer1 = sigmoid(np.dot(layer0, weights0))  # (1,5)
        layer2 = sigmoid(np.dot(layer1, weights1))  # (1,1)

        # --- BACKWARD PASS ---
        layer2_error = y - layer2
        # delta(L) = 2(a(L)-y) * sigmoid_derivative(z(L))
        layer2_delta = layer2_error * sigmoid_derivative(layer2)
        # layer1_error = w(L) * delta(l)
        layer1_error = layer2_delta.dot(weights1.T)
        # delta(L-1) = sigmoid_derivative(z(L-1)) * w(L) * delta(l)
        layer1_delta = layer1_error * sigmoid_derivative(layer1)

        # --- UPDATE WEIGHTS ---
        # Since batch=1, no averaging; scale by learning_rate
        # Gradients = layer1.T.dot(layer2_delta)
        weights1 += learning_rate * layer1.T.dot(layer2_delta)
        weights0 += learning_rate * layer0.T.dot(layer1_delta)


# --- 5. Predict Function (Same as Before, but for Single Sentence) ---
def predict(sentence):
    print(f"Predicting for sentence: '{sentence}'")
    # Convert to vector (1x12)
    vector = sentence_to_vector(sentence, vocabulary)
    layer0 = vector.reshape(1, -1)

    # Forward pass using current weights
    layer1 = sigmoid(np.dot(layer0, weights0))
    layer2 = sigmoid(np.dot(layer1, weights1))

    print(f"Spam probability: {layer2[0, 0]:.2f}")
    if layer2[0, 0] > 0.5:
        print("Result: This is SPAM.\n")
    else:
        print("Result: This is NOT SPAM.\n")


# --- Example Usage: Train on Original 4 Examples One by One ---
# You can call train_on_example multiple times to train incrementally.
# Here, train on each for 5000 iterations to simulate learning (adjust as needed).


print("Training on second example...")
train_on_example("urgent action required, verify your account", 1, iterations=5000)

print("Training on first example...")
train_on_example("hello, let's schedule a meeting", 0, iterations=5000)


print("Training on third example...")
train_on_example("buy now, free money for every winner", 1, iterations=5000)

print("Training on fourth example...")
train_on_example("hello, meeting about your account", 0, iterations=5000)

print("Training finished on initial examples.\n")

# Test predictions after initial training
predict("urgent! claim your free money now")
predict("hello, let's have a meeting about the account")

# --- To Train on More Examples Later ---
# Just call the function again with new data
print("Adding and training on a new spam example...")
train_on_example("winner of free action now", 1, iterations=5000)

print("Adding and training on a new not-spam example...")
train_on_example("schedule your meeting", 0, iterations=5000)

# Test again to see improvement
predict("urgent! claim your free money now")
predict("hello, let's have a meeting about the account")