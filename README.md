# Library Oracle

A machine learning project to predict book popularity in a high school library, created for a high school science project.

## Project Structure

- `data_generation.py`: Script to generate a synthetic dataset of books with metadata and borrowing history
- `data_preparation.py`: Script to process and prepare data for model training
- `model_training.py`: Script to train and evaluate the prediction model
- `make_prediction.py`: Script to load the trained model, preprocess new data, and make predictions
- `datasets/`: Directory containing generated and processed datasets
- `models/`: Directory for storing trained models and visualizations
- `notebooks/`: Jupyter notebooks for data exploration and visualization
- `flask_api.py`: Flask API to serve predictions for book popularity

## Setup

### Setting up a Virtual Environment

This project requires Python 3.8 or newer. You must use a virtual environment to avoid conflicts with your system Python:

#### On macOS:
```bash
# Install virtualenv if not already installed
pip3 install --user virtualenv

# Create a virtual environment
python3 -m virtualenv venv

# Activate the virtual environment
source venv/bin/activate
```

#### On Linux:
```bash
# Install virtualenv if not already installed
pip3 install --user virtualenv

# Create a virtual environment
python3 -m virtualenv venv

# Activate the virtual environment
source venv/bin/activate
```

#### On Windows:
```bash
# Install virtualenv if not already installed
pip3 install --user virtualenv

# Create a virtual environment
python3 -m virtualenv venv

# Activate the virtual environment
venv\Scripts\activate
```

### Troubleshooting Python Setup

If you encounter a "ModuleNotFoundError: No module named 'distutils'" error:

#### On macOS:
Install the Python development package:
```bash
brew install python-setuptools python3-dev
```

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-distutils python3-dev python3-setuptools
```

#### On Windows:
Python distutils should be included with the standard installation, but if missing, reinstall Python with the "Customize installation" option and ensure "pip" and "py launcher" are selected.

### Installing Dependencies

Once your virtual environment is activated (you should see `(venv)` at the beginning of your command prompt), install the required packages:

```bash
# Install dependencies
pip3 install -r requirements.txt
```

If you plan to run the Flask API, ensure Flask is installed:

```bash
pip3 install flask
```

### Running the Project

```bash
# Generate synthetic dataset
python3 data_generation.py

# Prepare data for training
python3 data_preparation.py

# Train and evaluate model
python3 model_training.py

# Make predictions for new books
python3 make_prediction.py --input_file=sample_book.json
```

### Running the Flask API

To start the Flask API for making predictions:

```bash
python3 flask_api.py
```

Send a POST request to `http://127.0.0.1:5000/predict` with a JSON payload containing book data. Example:

```bash
curl -X POST -H "Content-Type: application/json" -d @sample_book.json http://127.0.0.1:5000/predict
```

### Using Jupyter Notebooks

To run the Jupyter notebooks for data visualization:

```bash
# Make sure your virtual environment is activated, then:
python3 -m jupyter notebook
```

Navigate to the `notebooks` directory and open `data_visualization.ipynb`.