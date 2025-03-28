#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_preprocessed_data():
    """Load the preprocessed data for model training"""
    try:
        X_train = pd.read_csv('datasets/X_train_preprocessed.csv')
        X_val = pd.read_csv('datasets/X_val_preprocessed.csv')
        X_test = pd.read_csv('datasets/X_test_preprocessed.csv')
        y_train = pd.read_csv('datasets/y_train.csv').iloc[:, 0]
        y_val = pd.read_csv('datasets/y_val.csv').iloc[:, 0]
        y_test = pd.read_csv('datasets/y_test.csv').iloc[:, 0]
        print("Successfully loaded preprocessed data.")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except FileNotFoundError:
        print("Preprocessed data files not found. Please run data_preparation.py first.")
        exit(1)

def train_models(X_train, y_train, X_val, y_val):
    """Train multiple regression models and evaluate on validation set"""
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Dictionary to store model results
    model_results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"{name} Validation Results:")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  R²: {val_r2:.4f}")
        
        # Store model and results
        model_results[name] = {
            'model': model,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'val_predictions': y_val_pred
        }
    
    return model_results

def select_best_model(model_results, X_test, y_test):
    """Select the best model based on validation results and evaluate on test set"""
    # Find the model with the best validation R² score
    best_model_name = max(model_results.items(), key=lambda x: x[1]['val_r2'])[0]
    best_model = model_results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    
    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Test Results for {best_model_name}:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    # Save the best model
    best_model_path = f"models/{best_model_name.lower().replace(' ', '_')}_model.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Best model saved to {best_model_path}")
    
    return best_model, best_model_name, y_test_pred

def analyze_feature_importance(best_model, best_model_name, X_train):
    """Analyze and visualize feature importance for tree-based models"""
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        # Get feature importances
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        
        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('models/feature_importance.png')
        
        # Save the feature importance data
        feature_importance_df.to_csv('models/feature_importance.csv', index=False)
        
        print("\nFeature importance analysis completed and saved.")
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10))
    else:
        print("\nFeature importance analysis is only available for tree-based models.")

def visualize_predictions(y_test, y_test_pred, best_model_name):
    """Create visualizations for model performance"""
    # Scatter plot of actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Popularity Score')
    plt.ylabel('Predicted Popularity Score')
    plt.title(f'Actual vs. Predicted Popularity - {best_model_name}')
    plt.tight_layout()
    plt.savefig('models/actual_vs_predicted.png')
    
    # Residual plot
    residuals = y_test - y_test_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.xlabel('Predicted Popularity Score')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {best_model_name}')
    plt.tight_layout()
    plt.savefig('models/residuals.png')
    
    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='k', linestyle='--', lw=2)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution - {best_model_name}')
    plt.tight_layout()
    plt.savefig('models/residual_distribution.png')
    
    print("\nPrediction visualizations saved to models/ directory.")

def preprocess_new_books():
    """Load and preprocess new books for prediction"""
    try:
        # Load new books
        new_books_df = pd.read_csv('datasets/new_books_for_prediction.csv')
        
        # Get the features as in the prepared data
        feature_cols = [
            'publication_year', 'page_count', 'is_series', 'has_movie_adaptation',
            'awards_count', 'average_goodreads_rating', 'has_ebook', 'has_audiobook',
            'copies_available', 'is_required_reading', 'is_teacher_recommended',
            'genre', 'reading_level', 'age_category'
        ]
        
        # Create a copy of the features subset
        features_df = new_books_df[feature_cols].copy()
        
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
        
        # Convert boolean columns to integers (0/1)
        bool_cols = ['is_series', 'has_movie_adaptation', 'has_ebook', 'has_audiobook', 
                    'is_required_reading', 'is_teacher_recommended']
        
        for col in bool_cols:
            features_df[col] = features_df[col].astype(int)
        
        # Load the preprocessing tools
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        # Identify categorical columns for encoding
        categorical_cols = ['genre', 'reading_level', 'age_category']
        numerical_cols = [col for col in feature_cols if col not in categorical_cols and col not in bool_cols]
        
        # Scale numerical features
        X_new_scaled_numerical = scaler.transform(features_df[numerical_cols])
        
        # Convert scaled arrays back to DataFrames
        X_new_scaled_numerical_df = pd.DataFrame(X_new_scaled_numerical, 
                                               columns=numerical_cols)
        
        # Encode categorical features
        X_new_encoded_categorical = encoder.transform(features_df[categorical_cols])
        
        # Get the new column names after one-hot encoding
        categorical_feature_names = []
        for i, col in enumerate(categorical_cols):
            # Get all possible categories for this column
            categories = encoder.categories_[i]
            # Create column names for each category
            categorical_feature_names.extend([f"{col}_{category}" for category in categories])
        
        # Convert encoded arrays to DataFrames
        X_new_encoded_categorical_df = pd.DataFrame(X_new_encoded_categorical, 
                                                  columns=categorical_feature_names)
        
        # Get boolean columns
        X_new_bool = features_df[bool_cols]
        
        # Combine preprocessed features
        X_new_preprocessed = pd.concat([X_new_scaled_numerical_df, X_new_encoded_categorical_df, X_new_bool], axis=1)
        
        print(f"Preprocessed {len(X_new_preprocessed)} new books for prediction.")
        
        return X_new_preprocessed, new_books_df
    except FileNotFoundError:
        print("New books file not found. Make sure to run data_preparation.py first.")
        exit(1)

def predict_new_books(best_model, X_new_preprocessed, new_books_df):
    """Predict popularity for new books"""
    # Predict popularity scores for new books
    predicted_scores = best_model.predict(X_new_preprocessed)
    
    # Add predictions to the new books DataFrame
    new_books_df['predicted_popularity_score'] = predicted_scores
    
    # Sort by predicted popularity score
    new_books_df = new_books_df.sort_values('predicted_popularity_score', ascending=False)
    
    # Save predictions to CSV
    new_books_df.to_csv('datasets/new_books_with_predictions.csv', index=False)
    
    # Display the predictions for the top books
    print("\nTop 5 Most Promising New Books:")
    top_new_books = new_books_df[['book_id', 'title', 'author', 'genre', 'predicted_popularity_score']].head(5)
    print(top_new_books)
    
    return new_books_df

def main():
    print("Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    print("\nTraining and evaluating models...")
    model_results = train_models(X_train, y_train, X_val, y_val)
    
    print("\nSelecting best model and evaluating on test set...")
    best_model, best_model_name, y_test_pred = select_best_model(model_results, X_test, y_test)
    
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(best_model, best_model_name, X_train)
    
    print("\nVisualizing model predictions...")
    visualize_predictions(y_test, y_test_pred, best_model_name)
    
    print("\nPreprocessing new books for prediction...")
    X_new_preprocessed, new_books_df = preprocess_new_books()
    
    print("\nPredicting popularity for new books...")
    predict_new_books(best_model, X_new_preprocessed, new_books_df)
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()