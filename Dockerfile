# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /App

# Copy the current directory contents into the container at /app
COPY . /App

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV NAME=env

# Add this before your CMD line
RUN mkdir -p /tmp/nltk_data && chmod 777 /tmp/nltk_data

# Run app.py when the container launches
CMD ["python", "App.py"]
