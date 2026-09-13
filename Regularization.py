# Regularization is a technique used to prevent a ML model from overfitting

def train_regularized_regression(X, y, lr=0.01, alpha=0.1, epochs=500, method='ridge'):
    w = 0.0
    b = 0.0
    n = len(X)

    for epoch in range(epochs):
        w_gradient = 0
        b_gradient = 0

        # Calculate the standard data error gradients
        for i in range(n):
            prediction = w * X[i] + b
            error = y[i] - prediction
            w_gradient = -2 * X[i] * error
            b_gradient = -2 * error

        w_gradient = w_gradient / n
        b_gradient = b_gradient / n

        #  APPLY THE REGULARIZATION TO THE PENALTY
        if method == 'ridge':
            # L2 Penalty derivative: 2 * alpha * w
            w_gradient += 2 * alpha * w

        elif method == 'lasso':
            #  L1 Penalty derivative: alpha * sign(w)
            if w > 0:
                w_gradient += alpha
            elif w < 0:
                w_gradient -= alpha

        # update weight and bias
        w = w - (lr * w_gradient)
        b = b - (lr * b_gradient)

    return w, b

# train data
X_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.2, 3.8, 6.1, 7.9, 10.2] 

# compare the results
w_none, b_none = train_regularized_regression(X_data, y_data, method="none")
w_ridge, b_ridge = train_regularized_regression(X_data, y_data, method="ridge", alpha=0.5)
w_lasso, b_lasso = train_regularized_regression(X_data, y_data, method="lasso", alpha=0.5)

print(f"Vanilla Linear Regression: y = {w_none:.4f}x + {b_none:.4f}")
print(f"Ridge Regression (L2):     y = {w_ridge:.4f}x + {b_ridge:.4f}  <-- (Weight is shrunk)")
print(f"Lasso Regression (L1):     y = {w_lasso:.4f}x + {b_lasso:.4f}  <-- (Weight is shrunk harder)")
