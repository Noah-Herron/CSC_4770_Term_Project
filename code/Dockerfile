FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install (create a requirements.txt in your repo)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Set a default environment variable for the node role
ENV NODE_ROLE=unassigned

# Execute the application
CMD ["python", "app.py"]