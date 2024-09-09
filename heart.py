import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
heart_data = pd.read_csv('/Users/saifshaik/Documents/projects/HeartPS/API/data.csv')

# Split data into features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter
model.fit(X_train_scaled, Y_train)

# Evaluate the model (optional but recommended)
Y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Make a prediction with input data
input_data = [63, 0, 3, 130, 240, 0, 1, 169, 0, 0.5, 1, 0, 3]
input_data_np = np.array(input_data).reshape(1, -1)
input_data_scaled = scaler.transform(input_data_np)  # Scale input data
prediction = model.predict(input_data_scaled)

# Print the prediction result
if prediction[0] == 0:
    print("The Person does not have Heart Disease")
else:
    print("The Person has Heart Disease")
