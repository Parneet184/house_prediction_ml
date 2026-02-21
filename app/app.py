# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# app = Flask(__name__)
# model = joblib.load('../model/model.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
    
#     location = data.get('location')
#     country = data.get('country')

#     if not location or not country:
#         return jsonify({'error': 'Missing location or country'}), 400

#     input_df = pd.DataFrame([{'location': location, 'country': country}])
#     prediction = model.predict(input_df)[0]

#     return jsonify({'predicted_price': round(prediction, 2)})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get absolute path to model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../model/model.pkl")

model = joblib.load(model_path)

@app.route("/")
def home():
    return "ML API is running successfully 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        location = data.get('location')
        country = data.get('country')

        if not location or not country:
            return jsonify({'error': 'Missing location or country'}), 400

        input_df = pd.DataFrame([{
            'location': location,
            'country': country
        }])

        prediction = model.predict(input_df)[0]

        return jsonify({
            'predicted_price': float(round(prediction, 2))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)