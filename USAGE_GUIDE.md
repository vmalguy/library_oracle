# Library Oracle - Usage Guide

This document provides detailed instructions on how to use the Library Oracle system to predict book popularity in your high school library.

## Project Overview

Library Oracle is a machine learning system designed to forecast the potential popularity of new books in a high school library setting. The system analyzes various book attributes (genre, topics, reading level, etc.) and predicts how popular a book might be with students, helping librarians and school administrators make informed acquisition decisions.

## Setup and Installation

### Prerequisites

- Python 3.8 or newer
- pip3 (Python 3 package installer)

### Installation

1. Clone or download this repository to your local machine
2. Install required packages:

```bash
pip3 install -r requirements.txt
```

## Using the System

### Step 1: Generate or Use a Dataset

If you don't have your own library dataset, you can generate a synthetic dataset using the data_generation.py script:

```bash
python3 data_generation.py
```

This will create:
- A dataset of 2,000 books with various attributes
- A dataset of 800 fictional students with different profiles
- A checkout history that simulates book borrowing patterns
- Popularity metrics for each book based on checkout patterns

The data will be saved in the `datasets/` directory.

### Step 2: Prepare the Data for Model Training

Run the data preparation script:

```bash
python3 data_preparation.py
```

This will:
- Process the raw data into features suitable for model training
- Split the data into training, validation, and test sets
- Apply feature scaling and encoding
- Generate a test set of "new books" for prediction testing

### Step 3: Train the Model

Run the model training script:

```bash
python3 model_training.py
```

This will:
- Train multiple regression models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting)
- Evaluate each model on the validation set
- Select the best-performing model
- Analyze feature importance to understand what book characteristics drive popularity
- Generate visualizations of model performance
- Apply the model to predict popularity for a test set of new books

### Step 4: Use the Model for Predictions

To predict the popularity of a new book, create a JSON file describing the book's attributes (see `sample_book.json` for an example), then run:

```bash
python3 make_prediction.py --input_file=your_book.json
```

## Understanding the Prediction Results

The prediction result is a popularity score on a scale of 0-100, where:
- 0-20: Low popularity (minimal student interest expected)
- 21-40: Below average popularity
- 41-60: Average popularity
- 61-80: Above average popularity
- 81-100: High popularity (significant student interest expected)

The model bases its predictions on several key factors, including:
- Book attributes (genre, reading level, page count)
- Content characteristics (main topics, whether it's part of a series)
- External factors (Goodreads ratings, movie adaptations)
- School-specific factors (teacher recommendations, required reading status)

## Customizing the Model

To customize the model for your specific library:

1. Replace the synthetic data with your actual library checkout data in CSV format
2. Modify the feature set in `data_preparation.py` if your data has different attributes
3. Adjust the hyperparameters in `model_training.py` to optimize for your specific case
4. Retrain the model with your customized configuration

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed correctly
2. Check that all file paths are correct for your environment
3. Verify that input data formats match the expected structure
4. For prediction problems, ensure your JSON book description has all required fields

For persistent issues, consult the error messages for specific information about what went wrong.