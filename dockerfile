# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y build-essential


RUN pip install cython
RUN pip install numpy
RUN pip install setuptools
RUN pip install fairseq


# Install the dependencies
RUN pip install -r requirements.txt

# Copy the entire app directory to the container
COPY . .

# Expose the port that the FastAPI application listens on
EXPOSE 8000

# Run the FastAPI application using uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
