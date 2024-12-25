FROM python:3.10-slim

# Install necessary packages and clean up in the same layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    vim \
    # Required for OpenCV 
    libopencv-dev \
    #  C/C++ compiler
    gcc \
    # PostgreSQL database development files
    libpq-dev \ 
    # graphical applications
    libxcb1 \
    libx11-dev \
    libxext6 \
    libxi6 \
    libxrender1 \
    v4l-utils ffmpeg \
    && rm -rf /var/lib/apt/lists/* 
    
# Copy the project code into the container
COPY . /app
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Update pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set the working directory for running the application
WORKDIR /app

# # Set environment variable for display
# ENV DISPLAY=${DISPLAY}

# # Configure volumes
# VOLUME ["/tmp/.X11-unix:/tmp/.X11-unix"]

# Set the entrypoint
CMD ["python3", "main.py", "--main_disp", "--process_disp"]
