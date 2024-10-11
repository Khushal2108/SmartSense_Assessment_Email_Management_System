# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install tk

# Copy the rest of your project into the container
COPY . .

# Expose port 8000 (if you're running a web-based chatbot UI)
EXPOSE 8000

# Define environment variable to avoid buffering logs
ENV PYTHONUNBUFFERED=1

# Run the chatbot application
CMD ["python", "./app/main.py"]
