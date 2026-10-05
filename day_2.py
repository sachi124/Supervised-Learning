import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display the datset information
print("Dataset Info----")
print(X.describe())
print("\n target classes:", data.target_names)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train K-NN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict and evaluate
y_pred = knn.predict(X_test)
print("Accuracy without scaling:", accuracy_score(y_pred, y_test))


# Apply the Minmax Scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data
X_train_scale, X_test_scale, y_trian_scale, y_test_scale = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scale, y_trian_scale)

y_pred_scaled = knn_scaled.predict(X_test_scale)
print("Accuracy with MIn-Max Scaling:", accuracy_score(y_pred_scaled, y_test_scale))

# Apply the StandardScaler
standard = StandardScaler()
X_stand = standard.fit_transform(X)

X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_stand, y, test_size=42, random_state=42)

knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_train_std, y_train_std)

Y_pred_std = knn_std.predict(X_test_std)
print("Accuracy with StandardScaler:", accuracy_score(Y_pred_std, y_test_std))





