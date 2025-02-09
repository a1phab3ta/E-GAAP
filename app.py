from flask import Flask, render_template, request, jsonify
import random
import csv
import io
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("geomagnetic_model.pkl")


def generate_random_values():
    return {
        "bz_gsm": round(random.uniform(-10, 10)),
        "bt": round(random.uniform(0, 20)),
        "density": round(random.uniform(0, 50)),
        "speed": round(random.uniform(200, 800)),
        "temperature": round(random.uniform(10000, 500000))
    }

def predict_storm(data):
    features = np.array([data["speed"], data["bt"], data["temperature"], data["bz_gsm"], data["density"]]).reshape(1, -1)
    
    prediction = model.predict(features)
    return prediction[0]

def interpret_prediction(prediction):
    if prediction >= 0:
        classification = "Quiet"
        effects = "No disturbance"
    elif -20 <= prediction < 0:
        classification = "Weak"
        effects = "Minor fluctuations"
    elif -50 <= prediction < -20:
        classification = "Moderate"
        effects = "Small disturbances in radio and GPS"
    elif -100 <= prediction < -50:
        classification = "Strong"
        effects = "Possible power grid & satellite effects"
    elif -200 <= prediction < -100:
        classification = "Severe"
        effects = "Widespread disruptions, auroras visible at lower latitudes"
    else:
        classification = "Extreme"
        effects = "Power grid failures, major satellite issues"
    
    return classification, effects

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        # Handle single row submission
        data = request.get_json()
        prediction = predict_storm(data)
        classification, effects = interpret_prediction(prediction)
        return jsonify({
            "prediction": prediction,
            "classification": classification,
            "effects": effects
        })
    else:
        # Handle CSV file upload
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read the CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.DictReader(stream)
        results = []
        
        for row in csv_reader:
            # Convert row values to floats
            row_data = {key: float(value) for key, value in row.items()}
            prediction = predict_storm(row_data)
            classification, effects = interpret_prediction(prediction)
            results.append({
                **row_data,
                "prediction": prediction,
                "classification": classification,
                "effects": effects
            })
        
        return jsonify(results)

@app.route('/randomize', methods=['GET'])
def randomize():
    return jsonify(generate_random_values())

if __name__ == '__main__':
    app.run(debug=True)