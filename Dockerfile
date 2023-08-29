# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN set -e \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn
    

# Expose port 8080 for the Flask app
EXPOSE 8080

# Run the Flask app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]



#sudo docker build -t final3 .
#sudo docker run -p 8080:8080 final3
