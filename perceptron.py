import numpy as np

class ClassicPerceptron:
    def __init__(self, input_size, lr=0.1):
        self.weight = np.zeros(input_size)
        self.bias = 0.0
        self.lr = lr 

    def step_function(self, Z):
        return 1 if Z >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weight) + self.bias
        return self.step_function(z)

    def train(self, X, y, epochs=10):
        print("Training Classic Perceptron....")
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction =self.predict(xi)
                error = target - prediction


                if error != 0:
                    self.weight += self.lr * error * xi
                    self.bias += self.lr * error
                    error += 1

            print(f"Epoch: {epoch+1}: Made {error} errors.")
            if errors == 0:
                print("Covered Perfectly!\n")
                break


# --- Testing the Beginner/Intermediate Level ---
# Dataset for an AND gate (Linearly Separable)
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_and = np.array([0,0,0,1])

perceptron = ClassicPerceptron(input_size=2)
perceptron.train(X_and, Y_and)

# Show the trained weights forming the linear boundary equation: w1*x1 + w2*x2 + b = 0
print(f"Final Weights: {perceptron.weight}, Final Bias: {perceptron.bias}")
print(f"Prediction for: {perceptron.predict(np.array([1, 1]))}\n")
