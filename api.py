from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load("fish_species_classifier.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = [float(request.form.get(key)) for key in ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]]
        data_scaled = scaler.transform([data])  
        prediction = model.predict(data_scaled) 
        predicted_species = le.inverse_transform(prediction)[0]  

        return jsonify({"prediction": predicted_species})  

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)