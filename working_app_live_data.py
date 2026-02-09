from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import urllib.request
import json
import requests
from datetime import datetime

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load PIN-to-location data
pin_data = pd.read_csv("PIN.csv", encoding="latin1")

# Visual Crossing API Key
API_KEY = "DCDVS6HGLC8S6F657B22M9NNM"

# Fetch weather data
def fetch_weather(lat, lon):
    today = datetime.today().strftime('%Y-%m-%d')
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{today}?unitGroup=metric&key={API_KEY}&contentType=json'
    try:
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode())
        current = data['days'][0]
        temp = current.get('temp', 0)
        humidity = current.get('humidity', 0)
        wind_speed = current.get('windspeed', 0)
        precipitation = current.get('precip', 0)
        return temp, humidity, wind_speed, precipitation
    except Exception as e:
        print("Weather API error:", e)
        return 0, 0, 0, 0

# Fetch NPK from ThingSpeak
def fetch_npk():
    try:
        url = 'https://api.thingspeak.com/channels/1942826/feeds/last.json'
        response = requests.get(url)
        data = response.json()
        N = float(data.get('field1') or 0)
        P = float(data.get('field2') or 0)
        K = float(data.get('field3') or 0)
        return N, P, K
    except Exception as e:
        print("ThingSpeak error:", e)
        return 0, 0, 0

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        pincode = request.form["pincode"]
        land_size = float(request.form["land_size"])

        if int(pincode) in pin_data["Pincode"].values:
            row = pin_data[pin_data["Pincode"] == int(pincode)].iloc[0]
            lat, lon = row["Latitude"], row["Longitude"]

            temp, humidity, wind_speed, precipitation = fetch_weather(lat, lon)
            estimated_moisture = round(0.6 * humidity + 0.3 * precipitation - 0.1 * wind_speed, 2)

            N, P, K = fetch_npk()

            features = pd.DataFrame([[lat, lon, temp, humidity, N, P, K]],
                columns=["Latitude", "Longitude", "Avg_temp", "Avg_humidity", "N", "P", "K"])
            pred = model.predict(features)
            crop = label_encoder.inverse_transform(pred)[0]

            result = {
                "crop": crop,
                "temp": temp,
                "humidity": humidity,
                "moisture": estimated_moisture,
                "N": N,
                "P": P,
                "K": K
            }
        else:
            result = "Invalid pincode."

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>🌱 AI Crop Recommendation</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(to right, #e0f7fa, #f0f4c3);
                font-family: 'Segoe UI', sans-serif;
                padding: 40px 0;
            }
            .card-glass {
                backdrop-filter: blur(14px);
                background-color: rgba(255, 255, 255, 0.6);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
                border-radius: 20px;
                padding: 30px;
                transition: transform 0.3s ease;
            }
            .card-glass:hover {
                transform: scale(1.02);
            }
            .section-title {
                font-weight: bold;
                font-size: 1.5rem;
                margin-bottom: 20px;
            }
            .highlight {
                font-size: 1.25rem;
                font-weight: 600;
                color: #2e7d32;
            }
            .icon {
                font-size: 1.2rem;
                margin-right: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="text-center mb-5">
                <h1 class="fw-bold">🌾 AI Crop Recommender</h1>
                <p class="text-muted">Get the best crop for your land using live weather & soil data</p>
            </div>

            <div class="card-glass mb-5">
                <form method="post">
                    <div class="mb-3">
                        <label class="form-label">📍 Pincode</label>
                        <input type="text" class="form-control" name="pincode" placeholder="Enter your location's pincode" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">🌿 Land Size (acres)</label>
                        <input type="number" class="form-control" name="land_size" step="0.1" placeholder="E.g. 1.5" required>
                    </div>
                    <button type="submit" class="btn btn-success w-100">🔍 Predict Best Crop</button>
                </form>
            </div>

            {% if result %}
                {% if result == 'Invalid pincode.' %}
                    <div class="alert alert-danger text-center">❌ Invalid Pincode entered.</div>
                {% else %}
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="card-glass">
                                <div class="section-title">🌤️ Weather Insights</div>
                                <p><span class="icon">🌡️</span>Temperature: <span class="highlight">{{ result.temp }} °C</span></p>
                                <p><span class="icon">💧</span>Humidity: <span class="highlight">{{ result.humidity }} %</span></p>
                                <p><span class="icon">🌱</span>Estimated Soil Moisture: <span class="highlight">{{ result.moisture }} %</span></p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card-glass">
                                <div class="section-title">🧪 NPK Nutrient Levels</div>
                                <p><span class="icon">🧬</span>Nitrogen (N): <span class="highlight">{{ result.N }}</span></p>
                                <p><span class="icon">🧬</span>Phosphorus (P): <span class="highlight">{{ result.P }}</span></p>
                                <p><span class="icon">🧬</span>Potassium (K): <span class="highlight">{{ result.K }}</span></p>
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="card-glass mt-4 text-center">
                                <h4>🌾 Recommended Crop:</h4>
                                <p class="highlight display-6">{{ result.crop }}</p>
                            </div>
                        </div>
                    </div>
                {% endif %}
            {% endif %}
        </div>
    </body>
    </html>
    """, result=result)

if __name__ == "__main__":
    app.run(debug=False, port=5052)
