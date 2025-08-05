import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# --- 1. Define the "Unknown" Function and Generate Data ---
# In a real-world scenario, you wouldn't know this function.
# You would just have the x_train and y_train data.
# We define it here to generate sample data for our model to learn from.
# YOU CAN CHANGE THIS FUNCTION TO ANYTHING YOU LIKE!
def true_function(x):
    """The real function we want the model to learn."""
    # Example: A quadratic function
    return 0.5 * x**2 - 2 * x + 1
    # Example: A linear function
    # return 3 * x + 5
    # Example: A cubic function
    # return 0.1 * x**3 - 2 * x**2 + 5*x - 4

# Generate training data based on the true function
# np.linspace creates evenly spaced numbers over a specified interval.
x_train = np.linspace(-10, 10, 100)

# Add some random "noise" to the y-values to make it more realistic.
# Real-world data is rarely perfect.
np.random.seed(0) # for reproducibility
noise = np.random.normal(0, 5, x_train.shape) # Generate random noise
y_train = true_function(x_train) + noise

# The model expects X to be a 2D array, so we reshape it.
# The -1 means "calculate the appropriate number of rows" and 1 means "one column".
X_train_reshaped = x_train.reshape(-1, 1)


# --- 2. Create and Train the Machine Learning Model ---

# We'll use a Pipeline to chain together two steps:
# a) PolynomialFeatures: This transforms our input `x` into `[1, x, x^2, ...]`.
#    The 'degree' is a hyperparameter you can tune. For f(x)=x^2, a degree of 2 is perfect.
#    If the function is more complex, you might need a higher degree.
# b) LinearRegression: This model finds the best coefficients for the polynomial terms.

degree = 2 # Set the polynomial degree to learn
print(f"--- Model Setup: Trying to learn a polynomial of degree {degree} ---")

# Create the pipeline
pipeline = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=degree, include_bias=False)),
    ("linear_regression", LinearRegression())
])

# Train the model on our data
# The .fit() method is where the learning happens.
pipeline.fit(X_train_reshaped, y_train)


# --- 3. Evaluate the Model ---

# Let's see how well the model learned the training data.
y_train_pred = pipeline.predict(X_train_reshaped)
mse = mean_squared_error(y_train, y_train_pred)
print(f"\nModel Evaluation:")
print(f"Mean Squared Error on training data: {mse:.4f}")
print("(Lower is better. This tells us how far off, on average, our predictions are.)")


# --- 4. Predict New Values ---

# This is the core of your request: predict y for new x values.
print("\n--- Making New Predictions ---")
x_new = np.array([-7.5, 0.0, 8.5, 12.0]) # Some new x values we haven't seen before
X_new_reshaped = x_new.reshape(-1, 1)

# Use the trained pipeline to predict the y values
y_new_pred = pipeline.predict(X_new_reshaped)

for i in range(len(x_new)):
    print(f"For a new x = {x_new[i]:.1f}, the predicted y is: {y_new_pred[i]:.4f}")


# --- 5. Visualize the Results ---

# A plot helps us understand what the model has learned.
plt.figure(figsize=(12, 8))

# Plot the original training data (the noisy blue dots)
plt.scatter(x_train, y_train, label="Original Training Data (with noise)", alpha=0.6)

# Plot the true function (the black line we were trying to learn)
plt.plot(x_train, true_function(x_train), color='black', linestyle='--', linewidth=2, label="True Function")

# Plot the learned model's predictions (the solid red line)
# We use the trained model to predict y for a smooth range of x values.
x_plot = np.linspace(-12, 12, 200).reshape(-1, 1)
y_plot = pipeline.predict(x_plot)
plt.plot(x_plot, y_plot, color='red', linewidth=3, label=f"Learned Model (Degree {degree})")

# Plot the new predictions we just made (the large green circles)
plt.scatter(x_new, y_new_pred, color='green', s=150, zorder=5, label="New Predictions")

# Add labels and title for clarity
plt.title("Polynomial Regression: Learning f(x) from Data")
plt.xlabel("x value")
plt.ylabel("y value")
plt.legend()
plt.grid(True)
plt.show()
