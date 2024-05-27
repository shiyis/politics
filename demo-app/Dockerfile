# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install virtualenv
RUN apt-get update && apt-get install -y python3-venv && \
    python3 -m venv $VIRTUAL_ENV

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Activate the virtual environment
RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

EXPOSE 8050

# Define the command to run Gunicorn with your Dash application
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8050","app:server"]