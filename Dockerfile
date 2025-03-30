# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask globally to ensure the `flask` command is available
RUN pip install flask

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=flask_api.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the application
CMD ["python", "-m", "flask", "run"]
