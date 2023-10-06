# Use a Python base image for multi-architecture support
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9 

# Set the working directory
#WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the requirements
RUN pip3 install -r requirements.txt

# Copy the application code
COPY ./app /app

# Copy the models
COPY ./models /models

# Specify the command to run your FastAPI application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]
