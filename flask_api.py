from flask import Flask, request, jsonify, render_template_string
import json  # Import json for safe parsing
from make_prediction import load_model, preprocess_new_book, predict_popularity, predict_copies_needed  # Add predict_copies_needed

app = Flask(__name__)

# Load model and preprocessing tools
MODEL_PATH = 'models/ridge_regression_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/encoder.pkl'

model = load_model(MODEL_PATH)

# Enhanced HTML template for the UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Book Popularity Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
            background-color: #f4f4f9;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        form {
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: 20px auto;
        }
        label {
            font-weight: bold;
            margin-top: 10px;
            display: block;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background-color: #007BFF;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            display: block;
            margin: 20px auto;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-bottom: 20px;
            padding: 20px;
            background: #e9ffe9;
            border: 1px solid #b2d8b2;
            border-radius: 8px;
            font-size: 16px;
            color: #333;
            max-width: 600px;
            margin: 20px auto;
        }
        .mandatory {
            color: red;
        }
        .jaguar-image {
            display: block;
            margin: 0 auto 20px auto;
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .image-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .image-container img {
            max-width: 200px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .school-info {
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: 20px auto;
            text-align: center;
        }
        .school-info h2 {
            color: #333;
        }
        .school-info p {
            font-size: 14px;
            color: #555;
        }
        .school-info a {
            color: #007BFF;
            text-decoration: none;
        }
        .school-info a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Book Popularity Prediction</h1>
    <div class="image-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/Jaguar_closeup.jpg" alt="Jaguar Closeup" class="jaguar-image">
        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6e/Jaguar_resting.jpg" alt="Jaguar Resting" class="jaguar-image">
        <img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/Jaguar_in_forest.jpg" alt="Jaguar in Forest" class="jaguar-image">
    </div>
    <div class="school-info">
        <h2>About Jasper High School</h2>
        <p>Jasper High School, located in Plano, Texas, is known for its commitment to academic excellence and student success. The school offers a wide range of programs and extracurricular activities to support student growth and development.</p>
        <p>Learn more about Jasper High School by visiting their official website: <a href="https://www.pisd.edu/domain/1463" target="_blank">Jasper High School</a>.</p>
    </div>
    {% if prediction %}
        <div class="result">
            <h2>Prediction Result:</h2>
            <p>The book "<strong>{{ prediction.title }}</strong>" (ID: {{ prediction.book_id }}) is predicted to have a popularity score of <strong>{{ prediction.predicted_popularity_score }}</strong>/100.</p>
            <p>To ensure no more than 10 students are waiting to borrow this book at any given time, you should have at least <strong>{{ prediction.copies_needed }}</strong> copies available.</p>
        </div>
    {% endif %}
    <form method="POST" action="/predict-ui">
        <label for="book_id">Book ID: <span class="mandatory">*</span></label>
        <input type="text" id="book_id" name="book_id" required>

        <label for="title">Title: <span class="mandatory">*</span></label>
        <input type="text" id="title" name="title" required>

        <label for="publication_year">Publication Year: <span class="mandatory">*</span></label>
        <input type="number" id="publication_year" name="publication_year" required>

        <label for="page_count">Page Count: <span class="mandatory">*</span></label>
        <input type="number" id="page_count" name="page_count" required>

        <label for="genre">Genre: <span class="mandatory">*</span></label>
        <input type="text" id="genre" name="genre" required>

        <label for="average_goodreads_rating">Average Goodreads Rating:</label>
        <input type="number" step="0.1" id="average_goodreads_rating" name="average_goodreads_rating">

        <label for="copies_available">Copies Available:</label>
        <input type="number" id="copies_available" name="copies_available">

        <label for="is_series">Is Series:</label>
        <select id="is_series" name="is_series">
            <option value="false">No</option>
            <option value="true">Yes</option>
        </select>

        <label for="reading_level">Reading Level:</label>
        <input type="text" id="reading_level" name="reading_level">

        <label for="age_category">Age Category:</label>
        <input type="text" id="age_category" name="age_category">

        <label for="has_movie_adaptation">Has Movie Adaptation:</label>
        <select id="has_movie_adaptation" name="has_movie_adaptation">
            <option value="false">No</option>
            <option value="true">Yes</option>
        </select>

        <label for="has_diverse_characters">Has Diverse Characters:</label>
        <select id="has_diverse_characters" name="has_diverse_characters">
            <option value="false">No</option>
            <option value="true">Yes</option>
        </select>

        <label for="awards_count">Awards Count:</label>
        <input type="number" id="awards_count" name="awards_count">

        <label for="has_ebook">Has eBook:</label>
        <select id="has_ebook" name="has_ebook">
            <option value="false">No</option>
            <option value="true">Yes</option>
        </select>

        <label for="has_audiobook">Has Audiobook:</label>
        <select id="has_audiobook" name="has_audiobook">
            <option value="false">No</option>
            <option value="true">Yes</option>
        </select>

        <label for="copies_available">Copies Available:</label>
        <input type="number" id="copies_available" name="copies_available">

        <label for="is_required_reading">Is Required Reading:</label>
        <select id="is_required_reading" name="is_required_reading">
            <option value="false">No</option>
            <option value="true">Yes</option>
        </select>

        <label for="is_teacher_recommended">Is Teacher Recommended:</label>
        <select id="is_teacher_recommended" name="is_teacher_recommended">
            <option value="false">No</option>
            <option value="true">Yes</option>
        </select>

        <label for="main_topics">Main Topics (comma-separated):</label>
        <input type="text" id="main_topics" name="main_topics">

        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict-ui', methods=['POST'])
def predict_ui():
    if not model:
        return render_template_string(HTML_TEMPLATE, prediction="Error: Model not loaded")

    try:
        # Collect form data and convert to JSON-like dictionary
        book_data = {
            "book_id": request.form.get("book_id"),
            "title": request.form.get("title"),
            "author": request.form.get("author", None),
            "genre": request.form.get("genre"),
            "publisher": request.form.get("publisher", None),
            "publication_year": int(request.form.get("publication_year", 0)) if request.form.get("publication_year") else None,
            "page_count": int(request.form.get("page_count", 0)) if request.form.get("page_count") else None,
            "is_series": request.form.get("is_series") == "true",
            "reading_level": request.form.get("reading_level", None),
            "age_category": request.form.get("age_category", None),
            "has_movie_adaptation": request.form.get("has_movie_adaptation") == "true",
            "has_diverse_characters": request.form.get("has_diverse_characters") == "true",
            "awards_count": int(request.form.get("awards_count", 0)) if request.form.get("awards_count") else 0,
            "average_goodreads_rating": float(request.form.get("average_goodreads_rating", 0)) if request.form.get("average_goodreads_rating") else None,
            "has_ebook": request.form.get("has_ebook") == "true",
            "has_audiobook": request.form.get("has_audiobook") == "true",
            "copies_available": int(request.form.get("copies_available", 0)) if request.form.get("copies_available") else None,
            "is_required_reading": request.form.get("is_required_reading") == "true",
            "is_teacher_recommended": request.form.get("is_teacher_recommended") == "true",
            "main_topics": request.form.get("main_topics", "").split(",") if request.form.get("main_topics") else []
        }

        # Predict popularity
        predicted_score = predict_popularity(model, book_data, SCALER_PATH, ENCODER_PATH)
        
        # Predict copies needed
        copies_needed = predict_copies_needed(model, book_data, SCALER_PATH, ENCODER_PATH)

        # Format the prediction result
        response = {
            'book_id': book_data.get('book_id', 'unknown'),
            'title': book_data.get('title', 'unknown'),
            'predicted_popularity_score': float(predicted_score),
            'copies_needed': copies_needed
        }
        return render_template_string(HTML_TEMPLATE, prediction=response)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

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
        
        # Predict copies needed
        copies_needed = predict_copies_needed(model, book_data, SCALER_PATH, ENCODER_PATH)

        # Return the prediction
        response = {
            'book_id': book_data.get('book_id', 'unknown'),
            'title': book_data.get('title', 'unknown'),
            'predicted_popularity_score': float(predicted_score),
            'copies_needed': copies_needed
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
