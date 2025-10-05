import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Load trained model and label encoder
model = pickle.load(open('models/rfmodel.pkl', 'rb'))
label_encoder = pickle.load(open('models/encoder.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/predictcrop', methods=['GET', 'POST'])
def predict_datapoint():
    prediction = None  # default value
    if request.method == 'POST':
        try:
            N = float(request.form.get('N'))
            P = float(request.form.get('P'))
            K = float(request.form.get('K'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            ph = float(request.form.get('ph'))
            rainfall = float(request.form.get('rainfall'))

            # Predict using DataFrame with proper column names
            features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            prediction_array = model.predict(features)
            prediction = label_encoder.inverse_transform(prediction_array)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)



if __name__ == '__main__':
    app.run(host="0.0.0.0")
