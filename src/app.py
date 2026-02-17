from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BrentOilPrices.csv')
EVENTS_PATH = os.path.join(BASE_DIR, 'events.csv')
RESULTS_PATH = os.path.join(BASE_DIR, 'data', 'model_results.json')

@app.route('/api/prices', methods=['GET'])
def get_prices():
    try:
        if not os.path.exists(DATA_PATH):
             return jsonify({"error": "Data file not found"}), 404
             
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        # Return last 1000 days for performance
        df_tail = df.sort_values('Date').tail(1000)
        data = df_tail.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/events', methods=['GET'])
def get_events():
    try:
        if not os.path.exists(EVENTS_PATH):
            return jsonify({"error": "Events file not found"}), 404
            
        df = pd.read_csv(EVENTS_PATH)
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/change-points', methods=['GET'])
def get_change_points():
    try:
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({"error": "Model results not found. Running model first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
