from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load data and train model when the app starts
heart_data = pd.read_csv('data.csv')

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
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# Evaluate the model
Y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

@app.route('/')
def index():
    return "Heart Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.json
    
    # Convert input data to numpy array and scale it
    input_data = np.array(list(data.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    
    # Return the result as JSON
    result = {
        "prediction": int(prediction[0]),
        "message": "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have Heart Disease"
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
