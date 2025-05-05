import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
pickle_in = open("house_model.pkl", "rb")
classifier = pickle.load(pickle_in)

# Route for serving the homepage (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the prediction request
@app.route('/predict', methods=["GET"])
def predict():
    """
    Predict house prices based on user inputs from the form.
    Expected inputs: ['MedInc', 'HouseAge', 'Population', 'AveOccup']
    """
    input_cols = ['MedInc', 'HouseAge', 'Population', 'AveOccup']
    user_input = []

    for col in input_cols:
        value = request.args.get(col)
        user_input.append(float(value))  # Convert input values to floats
    
    prediction = classifier.predict([user_input])  # Predict using the model
    return str(prediction[0])  # Send back the prediction as a string

# Route for handling file upload predictions (optional)
@app.route('/predict_file', methods=["POST"])
def predict_file():
    # Predict based on file input (CSV format)
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return jsonify(predictions=list(prediction))

if __name__ == '__main__':
    # Run the application in debug mode
    app.run(debug=True)
