# Use official Python slim image for small size and AMD64 compatibility
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy the entire project contents into the Docker image
COPY . .

# Upgrade pip and install only required Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pandas \
        numpy \
        scikit-learn \
        joblib \
        spacy==3.7.2 \
        sentence-transformers \
        PyMuPDF \
        tqdm

# Link the spaCy model manually (already copied inside models/)
# This allows spaCy to find en_core_web_sm by name
RUN python -m spacy link models/en_core_web_sm/en_core_web_sm-3.8.0 en_core_web_sm

# Set the default command to run your application
CMD ["python", "main.py"]

