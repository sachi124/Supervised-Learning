from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# Load the dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(df.info())

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix:")
plt.show()

# Select feature with high correlation to the target
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
print("Feature with most correlated with target:")
print(correlated_features)

# Separate the feature and target
X = df.drop(columns=['target'])
y = df['target']

# Calculate the Mutual Information
mutual_info = mutual_info_regression(X, y)

# Create a dataframe for better visualization
mi_df = pd.DataFrame({'features': X.columns, 'Mutual Information': mutual_info})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

print("Mutual Information Scores:")
print(mi_df)