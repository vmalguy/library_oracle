from flask import Flask, request, jsonify
from make_prediction import load_model, preprocess_new_book, predict_popularity

app = Flask(__name__)

# Load model and preprocessing tools
MODEL_PATH = 'models/ridge_regression_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/encoder.pkl'

model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get JSON data from the request
        book_data = request.get_json()
        if not book_data:
            return jsonify({'error': 'Invalid input. JSON data is required.'}), 400

        # Predict popularity
        predicted_score = predict_popularity(model, book_data, SCALER_PATH, ENCODER_PATH)

        # Return the prediction
        response = {
            'book_id': book_data.get('book_id', 'unknown'),
            'title': book_data.get('title', 'unknown'),
            'predicted_popularity_score': float(predicted_score)
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
