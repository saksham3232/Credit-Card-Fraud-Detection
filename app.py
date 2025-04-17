from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Load scaler and model from models/ folder
scaler_path = os.path.join('models', 'scaler.pkl')
model_path = os.path.join('models', 'model.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect the input features as a single string
        input_data = request.form['features']
        
        # Convert the input string into a list of floats
        features = list(map(float, input_data.split(',')))

        # Ensure that there are exactly 30 features
        if len(features) != 30:
            return "Error: Please provide exactly 30 features (comma-separated)."

        # Reshape features to match the model's input format
        input_data_scaled = np.array(features).reshape(1, -1)

        # Scale the input data using the scaler
        input_data_scaled = scaler.transform(input_data_scaled)

        # Make prediction using the trained model
        prediction = model.predict(input_data_scaled)

        # Return the result based on prediction
        if prediction == 1:
            result = 'Fraudulent Transaction'
        else:
            result = 'Legitimate Transaction'

    except Exception as e:
        return f"Error in prediction: {e}"

    return render_template('index.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
