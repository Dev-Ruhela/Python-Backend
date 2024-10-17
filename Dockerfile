# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Copy the rest of your application code into the container
COPY . .

# Expose the port your app runs on
EXPOSE 7860

# Command to run your application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
