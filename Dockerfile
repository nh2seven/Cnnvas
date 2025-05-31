# Use official PyTorch image with CUDA 12.8 and cuDNN 9
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
COPY app.py run.py config.yaml setup.sh ./
COPY models/ ./models/
COPY utils/ ./utils/

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN chmod +x setup.sh && ./setup.sh

# Expose Gradio's default port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"]
