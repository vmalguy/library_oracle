#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import pickle

# Create datasets directory if it doesn't exist
os.makedirs('datasets', exist_ok=True)

def load_data():
    """Load the generated dataset files"""
    try:
        books_df = pd.read_csv('datasets/books.csv')
        students_df = pd.read_csv('datasets/students.csv')
        checkouts_df = pd.read_csv('datasets/checkouts.csv')
        books_with_metrics_df = pd.read_csv('datasets/books_with_metrics.csv')
        print("Successfully loaded dataset files.")
        return books_df, students_df, checkouts_df, books_with_metrics_df
    except FileNotFoundError:
        print("Dataset files not found. Please run data_generation.py first.")
        exit(1)

def prepare_features(books_df):
    """Prepare feature set for model training"""
    # Select relevant columns for prediction model
    feature_cols = [
        'publication_year', 'page_count', 'is_series', 'has_movie_adaptation',
        'awards_count', 'average_goodreads_rating', 'has_ebook', 'has_audiobook',
        'copies_available', 'is_required_reading', 'is_teacher_recommended',
        'genre', 'reading_level', 'age_category'
    ]
    
    # Create a copy of the features subset
    features_df = books_df[feature_cols].copy()
    
    # Handle missing values
    features_df.fillna({
        'is_series': False,
        'has_movie_adaptation': False,
        'awards_count': 0,
        'has_ebook': False,
        'has_audiobook': False,
        'is_required_reading': False,
        'is_teacher_recommended': False
    }, inplace=True)
    
    # Ensure target variable has no NaN values
    target = books_df['popularity_score'].fillna(0)
    
    # Convert boolean columns to integers (0/1)
    bool_cols = ['is_series', 'has_movie_adaptation', 'has_ebook', 'has_audiobook', 
                'is_required_reading', 'is_teacher_recommended']
    
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)
    
    # Identify categorical columns for encoding
    categorical_cols = ['genre', 'reading_level', 'age_category']
    numerical_cols = [col for col in feature_cols if col not in categorical_cols and col not in bool_cols]
    
    return features_df, target, numerical_cols, categorical_cols, bool_cols

def train_test_validation_split(features, target):
    """Split data into train, test, and validation sets"""
    # First split: separate out the test set (20% of data)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Second split: separate training and validation sets (validation is 25% of the remaining data = 20% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_val, X_test, numerical_cols, categorical_cols):
    """Preprocess the data for modeling"""
    # Initialize transformers
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Scale numerical features
    X_train_scaled_numerical = scaler.fit_transform(X_train[numerical_cols])
    X_val_scaled_numerical = scaler.transform(X_val[numerical_cols])
    X_test_scaled_numerical = scaler.transform(X_test[numerical_cols])
    
    # Convert scaled arrays back to DataFrames
    X_train_scaled_numerical_df = pd.DataFrame(X_train_scaled_numerical, 
                                             index=X_train.index, 
                                             columns=numerical_cols)
    X_val_scaled_numerical_df = pd.DataFrame(X_val_scaled_numerical, 
                                           index=X_val.index, 
                                           columns=numerical_cols)
    X_test_scaled_numerical_df = pd.DataFrame(X_test_scaled_numerical, 
                                            index=X_test.index, 
                                            columns=numerical_cols)
    
    # Encode categorical features
    X_train_encoded_categorical = encoder.fit_transform(X_train[categorical_cols])
    X_val_encoded_categorical = encoder.transform(X_val[categorical_cols])
    X_test_encoded_categorical = encoder.transform(X_test[categorical_cols])
    
    # Get the new column names after one-hot encoding
    categorical_feature_names = []
    for i, col in enumerate(categorical_cols):
        # Get all possible categories for this column
        categories = encoder.categories_[i]
        # Create column names for each category
        categorical_feature_names.extend([f"{col}_{category}" for category in categories])
    
    # Convert encoded arrays to DataFrames
    X_train_encoded_categorical_df = pd.DataFrame(X_train_encoded_categorical, 
                                                index=X_train.index, 
                                                columns=categorical_feature_names)
    X_val_encoded_categorical_df = pd.DataFrame(X_val_encoded_categorical, 
                                              index=X_val.index, 
                                              columns=categorical_feature_names)
    X_test_encoded_categorical_df = pd.DataFrame(X_test_encoded_categorical, 
                                               index=X_test.index, 
                                               columns=categorical_feature_names)
    
    # Get boolean columns
    bool_cols = [col for col in X_train.columns if col not in numerical_cols and col not in categorical_cols]
    X_train_bool = X_train[bool_cols]
    X_val_bool = X_val[bool_cols]
    X_test_bool = X_test[bool_cols]
    
    # Combine preprocessed features
    X_train_preprocessed = pd.concat([X_train_scaled_numerical_df, X_train_encoded_categorical_df, X_train_bool], axis=1)
    X_val_preprocessed = pd.concat([X_val_scaled_numerical_df, X_val_encoded_categorical_df, X_val_bool], axis=1)
    X_test_preprocessed = pd.concat([X_test_scaled_numerical_df, X_test_encoded_categorical_df, X_test_bool], axis=1)
    
    # Save the preprocessors for later use in prediction
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    print(f"Preprocessed data shape: {X_train_preprocessed.shape}")
    
    return X_train_preprocessed, X_val_preprocessed, X_test_preprocessed

def generate_book_prediction_dataset(books_df, students_df):
    """Create a special dataset for testing the popularity prediction of new books"""
    # Generate a small set of "new" books by modifying existing books
    new_books = []
    
    # Select 20 random books from the dataset
    sample_books = books_df.sample(20, random_state=42)
    
    # For each sample book, create a modified version as a "new" book
    for _, book in sample_books.iterrows():
        new_book = book.copy()
        new_book['book_id'] = f"NEW{book['book_id'][2:]}"  # Change the book ID
        new_book['title'] = f"New {book['title']}"  # Change the title
        
        # Modify some attributes to make it a "new" book
        new_book['publication_year'] = 2024  # Set publication year to next year
        new_book['acquisition_date'] = '2024-01-01'  # New acquisition date
        
        # Remove existing popularity metrics
        if 'total_checkouts' in new_book:
            new_book['total_checkouts'] = 0
        if 'popularity_score' in new_book:
            new_book['popularity_score'] = 0
        if 'recency_score' in new_book:
            new_book['recency_score'] = 0
        if 'avg_checkout_duration' in new_book:
            new_book['avg_checkout_duration'] = 0
        if 'checkouts_per_copy' in new_book:
            new_book['checkouts_per_copy'] = 0
        if 'checkouts_per_copy_score' in new_book:
            new_book['checkouts_per_copy_score'] = 0
            
        # Randomly modify some attributes to make the book different
        if np.random.random() < 0.3:
            new_book['has_movie_adaptation'] = not book['has_movie_adaptation']
        if np.random.random() < 0.3:
            new_book['average_goodreads_rating'] = min(5.0, book['average_goodreads_rating'] + np.random.uniform(-0.5, 0.5))
        if np.random.random() < 0.3:
            new_book['is_teacher_recommended'] = not book['is_teacher_recommended']
        
        new_books.append(new_book)
    
    # Create a DataFrame from the new books
    new_books_df = pd.DataFrame(new_books)
    
    # Save the new books to CSV for later reference
    new_books_df.to_csv('datasets/new_books_for_prediction.csv', index=False)
    
    print(f"Generated {len(new_books_df)} new books for prediction testing")
    
    return new_books_df

def main():
    print("Loading data...")
    books_df, students_df, checkouts_df, books_with_metrics_df = load_data()
    
    print("\nPreparing features...")
    features_df, target, numerical_cols, categorical_cols, bool_cols = prepare_features(books_with_metrics_df)
    
    print("\nSplitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(features_df, target)
    
    print("\nPreprocessing data...")
    X_train_preprocessed, X_val_preprocessed, X_test_preprocessed = preprocess_data(
        X_train, X_val, X_test, numerical_cols, categorical_cols
    )
    
    # Save preprocessed data for model training
    print("\nSaving preprocessed data...")
    X_train_preprocessed.to_csv('datasets/X_train_preprocessed.csv', index=False)
    X_val_preprocessed.to_csv('datasets/X_val_preprocessed.csv', index=False)
    X_test_preprocessed.to_csv('datasets/X_test_preprocessed.csv', index=False)
    y_train.to_csv('datasets/y_train.csv', index=False)
    y_val.to_csv('datasets/y_val.csv', index=False)
    y_test.to_csv('datasets/y_test.csv', index=False)
    
    # Generate and preprocess new books for prediction
    print("\nGenerating test dataset for new book predictions...")
    new_books_df = generate_book_prediction_dataset(books_df, students_df)
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()