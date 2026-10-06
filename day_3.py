import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display the dataset information
print("Dataset Info:")
print(df.info())

# preview the first few rows
print("\n Dataset Preview")
print(df.head())

# Apply one hot encoding
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
df_one_hot = df_one_hot.fillna(df_one_hot.median(numeric_only=True))

# Display encoded datasets
print("\n One Hot encoding")
print(df_one_hot.head())

# Apply the label encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# Display the label encoder
print("\n label Encoder")
print(df[['Pclass', 'Pclass_encoded']].head())

# Apply the frequency encoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Diplay the frequency
print("\n Frequency Encoding")
print(df[['Ticket', 'Ticket_frequency']].head())


X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df_one_hot['Survived']

# Split the datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# predict and evaluate
y_pred = model.predict(X_test)

print("The accuracy score with One_hot encoding is:", accuracy_score(y_test, y_pred))
