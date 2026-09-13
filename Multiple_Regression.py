def train_multiple_regression(X_matrix, y, lr=0.005, epochs=1000):
    """
    Trains Multiple Linear Regression using pure Python loops.
    X_matrix: A list of lists, where each sublist contains the features for one row.
    y: A list of target values.
    """
    num_samples = len(X_matrix)
    num_features = len(X_matrix[0])
    
    # 1. Initialize all feature weights and the bias to 0.0
    weights = [0.0] * num_features
    bias = 0.0
    
    for epoch in range(epochs):
        # Arrays to accumulate gradients for this epoch
        weight_gradients = [0.0] * num_features
        bias_gradient = 0.0
        
        # 2. Loop through every row of data
        for i in range(num_samples):
            # Calculate the prediction: w1*x1 + w2*x2 + ... + b
            prediction = 0.0
            for j in range(num_features):
                prediction += weights[j] * X_matrix[i][j]
            prediction += bias
            
            # Calculate the error for this data point
            error = y[i] - prediction
            
            # Accumulate gradients for each feature weight
            for j in range(num_features):
                weight_gradients[j] += -2 * X_matrix[i][j] * error
                
            # Accumulate gradient for bias
            bias_gradient += -2 * error
            
        # 3. Average the gradients across all samples and update weights
        for j in range(num_features):
            weight_gradients[j] /= num_samples
            weights[j] = weights[j] - (lr * weight_gradients[j])
            
        bias_gradient /= num_samples
        bias = bias - (lr * bias_gradient)
        
    return weights, bias

# --- Example Evaluation (Real Estate Use Case) ---
# Each row contains: [Square Footage (divided by 100), Number of Bedrooms]
X_data = [
    [10.0, 2],  # House 1
    [15.0, 3],  # House 2
    [20.0, 3],  # House 3
    [25.0, 4],  # House 4
    [30.0, 5]   # House 5
]
# Target: House price in thousands of dollars
y_data = [150, 220, 280, 360, 430]

# Train the model
final_weights, final_bias = train_multiple_regression(X_data, y_data, lr=0.002, epochs=5000)

print("--- Trained Model Results ---")
for i, w in enumerate(final_weights):
    print(f"Weight for Feature {i+1} (w{i+1}): {w:.4f}")
print(f"Bias (b): {final_bias:.4f}")

# --- Make a New Prediction ---
# Let's predict a house with 1,800 sq ft (18.0) and 3 bedrooms
new_house = [18.0, 3]
predicted_price = final_bias
for j in range(len(new_house)):
    predicted_price += final_weights[j] * new_house[j]

print(f"\nPredicted price for an 1800 sq ft, 3-bedroom house: ${predicted_price:.2f}k")



