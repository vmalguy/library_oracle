#!/usr/bin/env python3

import os
import argparse
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

"""
This script handles loading the model, preprocessing new data, and making predictions.
"""

def load_model(model_path):
    """Load a trained model from disk"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None

def preprocess_new_book(book_data, scaler_path, encoder_path):
    """Preprocess a new book for prediction"""
    # Load preprocessing tools
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    # Feature columns used in the model
    feature_cols = [
        'publication_year', 'page_count', 'is_series', 'has_movie_adaptation',
        'awards_count', 'average_goodreads_rating', 'has_ebook', 'has_audiobook',
        'copies_available', 'is_required_reading', 'is_teacher_recommended',
        'genre', 'reading_level', 'age_category'
    ]
    
    # Default values for missing attributes
    default_values = {
        'publication_year': 2000,
        'page_count': 100,
        'is_series': False,
        'has_movie_adaptation': False,
        'awards_count': 0,
        'average_goodreads_rating': 3.0,
        'has_ebook': False,
        'has_audiobook': False,
        'copies_available': 1,  # Default value for optional field
        'is_required_reading': False,
        'is_teacher_recommended': False,
        'genre': 'Unknown',
        'reading_level': 'Medium',
        'age_category': 'Adult'
    }
    
    # Fill missing attributes with default values
    for col in feature_cols:
        if col not in book_data or book_data[col] is None:
            book_data[col] = default_values[col]
    
    # Convert boolean columns to integers
    bool_cols = ['is_series', 'has_movie_adaptation', 'has_ebook', 'has_audiobook', 
                 'is_required_reading', 'is_teacher_recommended']
    for col in bool_cols:
        book_data[col] = int(book_data[col])
    
    # Identify categorical and numerical columns
    categorical_cols = ['genre', 'reading_level', 'age_category']
    numerical_cols = [col for col in feature_cols if col not in categorical_cols and col not in bool_cols]
    
    # Create DataFrames for each feature type
    book_df = pd.DataFrame([book_data])
    
    # Scale numerical features
    X_numerical = scaler.transform(book_df[numerical_cols])
    X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_cols)
    
    # Encode categorical features
    X_categorical = encoder.transform(book_df[categorical_cols])
    
    # Get the new column names after one-hot encoding
    categorical_feature_names = []
    for i, col in enumerate(categorical_cols):
        categories = encoder.categories_[i]
        categorical_feature_names.extend([f"{col}_{category}" for category in categories])
    
    X_categorical_df = pd.DataFrame(X_categorical, columns=categorical_feature_names)
    
    # Get boolean columns
    X_bool = book_df[bool_cols]
    
    # Combine preprocessed features
    X_preprocessed = pd.concat([X_numerical_df, X_categorical_df, X_bool], axis=1)
    
    return X_preprocessed

def predict_popularity(model, book_data, scaler_path, encoder_path):
    """Predict the popularity of a new book"""
    # Preprocess the book data
    X_preprocessed = preprocess_new_book(book_data, scaler_path, encoder_path)
    
    # Make prediction
    predicted_score = model.predict(X_preprocessed)[0]
    
    return predicted_score

def predict_copies_needed(model, book_data, scaler_path, encoder_path, demand_factor=10):
    """
    Predict the number of copies needed to ensure no more than 10 students are waiting for the book.
    """
    # Preprocess the book data
    X_preprocessed = preprocess_new_book(book_data, scaler_path, encoder_path)
    
    # Predict the popularity score
    predicted_score = model.predict(X_preprocessed)[0]
    
    # Calculate the number of copies needed
    # Assume demand_factor represents the number of students per popularity score unit
    copies_needed = max(1, int(predicted_score / demand_factor))
    
    return copies_needed

def main():
    """Main function to handle command line arguments and run prediction"""
    parser = argparse.ArgumentParser(description='Predict book popularity using trained model')
    parser.add_argument('--model_path', type=str, default='models/ridge_regression_model.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--scaler_path', type=str, default='models/scaler.pkl',
                        help='Path to the feature scaler file')
    parser.add_argument('--encoder_path', type=str, default='models/encoder.pkl',
                        help='Path to the categorical encoder file')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to JSON file containing book data')
    parser.add_argument('--output_file', type=str, default='prediction_result.json',
                        help='Path to save prediction results')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Load book data
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return
    
    try:
        # Read the input file as JSON
        book_data = pd.read_json(args.input_file, orient='records')
        if isinstance(book_data, pd.DataFrame) and len(book_data) > 0:
            book_data = book_data.iloc[0].to_dict()
        else:
            print("Invalid input file format. Expected a JSON object or array with at least one record.")
            return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Make prediction
    predicted_score = predict_popularity(model, book_data, args.scaler_path, args.encoder_path)
    
    # Prepare results
    result = {
        'book_id': book_data.get('book_id', 'unknown'),
        'title': book_data.get('title', 'unknown'),
        'predicted_popularity_score': float(predicted_score)
    }
    
    # Save results
    try:
        pd.DataFrame([result]).to_json(args.output_file, orient='records', indent=4)
        print(f"Prediction saved to {args.output_file}")
        print(f"Predicted popularity score: {predicted_score:.2f}")
    except Exception as e:
        print(f"Error saving prediction result: {e}")

def test_predict_popularity():
    """Test the predict_popularity function with mock data"""
    import numpy as np
    from unittest.mock import MagicMock

    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [42.0]

    # Mock book data
    mock_book_data = {
        'publication_year': 2020,
        'page_count': 300,
        'is_series': True,
        'has_movie_adaptation': False,
        'awards_count': 2,
        'average_goodreads_rating': 4.5,
        'has_ebook': True,
        'has_audiobook': False,
        'copies_available': 10,
        'is_required_reading': False,
        'is_teacher_recommended': True,
        'genre': 'fiction',
        'reading_level': 'medium',
        'age_category': 'teen'
    }

    # Mock preprocess_new_book to bypass file loading
    def mock_preprocess_new_book(book_data, scaler_path, encoder_path):
        numerical_features = np.array([[0.5, 0.3, 0.2]])  # Mock scaled numerical features
        categorical_features = np.array([[1, 0, 0]])  # Mock encoded categorical features
        boolean_features = np.array([[1, 0, 1, 0, 0, 1]])  # Mock boolean features
        return np.hstack([numerical_features, categorical_features, boolean_features])

    # Replace preprocess_new_book with the mock version
    global preprocess_new_book
    preprocess_new_book = mock_preprocess_new_book

    # Call predict_popularity
    predicted_score = predict_popularity(mock_model, mock_book_data, "mock_scaler_path", "mock_encoder_path")
    assert predicted_score == 42.0, f"Expected 42.0, got {predicted_score}"

if __name__ == "__main__":
    main()
    test_predict_popularity()