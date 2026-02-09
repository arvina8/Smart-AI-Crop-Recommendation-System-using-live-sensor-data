from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import urllib.request
import json
import requests
from datetime import datetime

app = Flask(__name__)

# Load required CSVs
pin_data = pd.read_csv('PIN.csv', encoding='latin1')
apc_data = pd.read_csv('APC.csv', encoding='latin1')

# Load updated model and label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Weather API Key
API_KEY = 'DCDVS6HGLC8S6F657B22M9NNM'

def fetch_weather_data(lat, lon, start_date):
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}?unitGroup=us&key={API_KEY}&contentType=json'
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        return json.loads(data.decode('utf-8'))
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def fetch_npk_from_public_thingspeak():
    try:
        url = 'https://api.thingspeak.com/channels/1388857/feeds/last.json'
        response = requests.get(url).json()

        N = float(response.get('field1') or 0)
        P = float(response.get('field2') or 0)
        K = float(response.get('field3') or 0)

        return N, P, K
    except Exception as e:
        print(f"Error fetching NPK data: {e}")
        return 0, 0, 0



@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crop Recommendation System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <h1 class="text-center">🌾 Crop Recommendation</h1>
            <form method="post" action="/predict" class="mt-4">
                <div class="mb-3">
                    <label for="pincode" class="form-label">📍 Enter Pincode</label>
                    <input type="text" id="pincode" name="pincode" class="form-control" required>
                </div>
                <div class="mb-3">
                    <label for="land_size" class="form-label">🌱 Land Size (in acres)</label>
                    <input type="number" step="0.1" id="land_size" name="land_size" class="form-control" required>
                </div>
                <button type="submit" class="btn btn-success w-100">Get Recommendation 🚀</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_pincode = request.form['pincode']
        land_size = float(request.form['land_size'])

        if user_pincode in pin_data['Pincode'].astype(str).values:
            row = pin_data[pin_data['Pincode'] == int(user_pincode)].iloc[0]
            latitude = row['Latitude']
            longitude = row['Longitude']
            place_name = row['Placename']
            district = row['District']
            state_name = row['StateName']

            # Fetch weather data
            start_date = datetime.today().strftime('%Y-%m-%d')
            weather_data = fetch_weather_data(latitude, longitude, start_date)

            if weather_data is None:
                return jsonify({"error": "Error fetching weather data."}), 500

            current_day = weather_data['days'][0]
            temperature = current_day.get('temp', 0)
            humidity = current_day.get('humidity', 0)

            # Fetch NPK values
            N, P, K = fetch_npk_from_public_thingspeak()

            # Include NPK in prediction
            features = pd.DataFrame([[latitude, longitude, temperature, humidity, N, P, K]],
            columns=["Latitude", "Longitude", "Avg_temp", "Avg_humidity", "N", "P", "K"])


            # Predict crop
            features = pd.DataFrame([[latitude, longitude, temperature, humidity, N, P, K]],
            columns=["Latitude", "Longitude", "Avg_temp", "Avg_humidity", "N", "P", "K"])


            predicted_index = model.predict(features)
            predicted_crop = label_encoder.inverse_transform(predicted_index)

            crop_data = apc_data[apc_data['Crop'] == predicted_crop[0]]
            average_yield = crop_data['Average_Yield'].values[0] if not crop_data.empty else 0
            estimated_production = land_size * average_yield

            return render_template_string('''
            <html><head><title>Crop Prediction</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet"></head>
            <body class="bg-light p-4">
                <div class="container">
                    <h1 class="text-center mb-4">🌿 Crop Recommendation Result</h1>
                    <div class="card p-4">
                        <h4>📍 Location Info</h4>
                        <p><strong>Pincode:</strong> {{ user_pincode }}</p>
                        <p><strong>Place:</strong> {{ place_name }}</p>
                        <p><strong>District:</strong> {{ district }}</p>
                        <p><strong>State:</strong> {{ state_name }}</p>
                        <h4 class="mt-4">🌦️ Weather</h4>
                        <p><strong>Temperature:</strong> {{ temperature }} °F</p>
                        <p><strong>Humidity:</strong> {{ humidity }}%</p>
                        <h4 class="mt-4">🧪 Soil Nutrients (Live)</h4>
                        <p><strong>Nitrogen (N):</strong> {{ N }}</p>
                        <p><strong>Phosphorus (P):</strong> {{ P }}</p>
                        <p><strong>Potassium (K):</strong> {{ K }}</p>

                        <h4 class="mt-4">🌱 Prediction</h4>
                        <p><strong>Recommended Crop:</strong> {{ predicted_crop }}</p>
                        <p><strong>Estimated Production:</strong> {{ estimated_production }} kg</p>
                    </div>
                    <div class="text-center mt-4">
                        <a href="/" class="btn btn-primary">🔁 Try Again</a>
                    </div>
                </div>
            </body></html>
            ''', user_pincode=user_pincode, place_name=place_name, district=district, state_name=state_name,
                 temperature=temperature, humidity=humidity, N=N, P=P, K=K,
                 predicted_crop=predicted_crop[0], estimated_production=estimated_production)

        else:
            return jsonify({"error": "Pincode not found."}), 404

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5052)
