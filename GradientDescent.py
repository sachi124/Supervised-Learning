def gradient_descent(X, y, learn_rate=0.01, epochs=1000):
    # Initialize the weight and bias to 0
    w = 0.00
    b = 0.00
    n = len(X)

    for epoch in range(epochs):
        # Initialize the gradient for this loop
        w_gradient = 0
        b_gradient = 0

        # Calculate the gradients by summing errors across all the data points
        for i in range(n):
            prediction = w * X[i] + b
            error = y[i] - prediction

            # Partial derivatives formula
            w_gradient += -2 * X[i] * error
            b_gradient += -2 * error

        # Average the gradients over all data samples
        w_gradient /= n
        b_gradient /= n

        # Update the parameter using learning rate
        w = w - (learn_rate * w_gradient)
        b = b - (learn_rate * b_gradient)

        # Optional: print progress every 100 iterations
        if epoch % 100 == 0:
            # Calculate the current Mean Square Error
            mse = sum(y[i] - (w * X[i] +b) ** 2 for i in range(n)) / n
            print(f"Epoch: {epoch}, Cost = {mse:.4f}, b = {b:.4f}")

    return w,b

X_data = [1, 2, 3, 4, 5]
y_data = [3.1, 4.9, 7.2, 8.8, 11.1]

best_w, best_b = gradient_descent(X_data, y_data, learn_rate=0.01, epochs=600)

print(f"\nFinal Trained Equation: y = {best_w:.2f}x + {best_b:.2f}")
