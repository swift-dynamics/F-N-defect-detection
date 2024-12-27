# Defect Detection System

## **Installation**

### **Step 1: Install System Dependencies**
Run the following command to install the required system libraries:
```bash
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    vim \
    libopencv-dev \  # Required for OpenCV 
    gcc \            # C/C++ compiler
    libpq-dev \      # PostgreSQL database development files
    libxcb1 \        # Graphical applications
    libx11-dev \
    libxext6 \
    libxi6 \
    libxrender1 \
    v4l-utils \      # Video tools
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*
```

---

### **Step 2: Set Up Python Environment**
1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Create ROI and Image Template**
To configure Regions of Interest (ROI) and image templates for defect detection:
```bash
python3 main.py --setting
```

---

### **Run Defect Detection**
To run the defect detection system for milk carton metallic detection and MFD/EXD text extraction:
```bash
python3 main.py --main_disp --process_disp --debug
```

#### **Available Flags**
| Flag               | Description                             |
|--------------------|-----------------------------------------|
| `--main_disp`      | Display the main GUI.                  |
| `--process_disp`   | Display the process GUI.               |
| `--debug`          | Enable debug logging.                  |
| `--save`           | Save detected images locally.          |
| `--minio`          | Upload detected images to MinIO.       |

---

## **Docker Support**

### **Build and Run All Services**
To build and start all services using Docker Compose:
```bash
docker compose up -d --build
```

---

### **Run Defect Detection Service Only**
To start only the defect detection service:
```bash
docker compose up -d fn-defect-detection
```

---

### **Save Alert Images to MinIO**
1. **Start MinIO services and create a bucket:**
   ```bash
   docker compose up -d minio minio-create-bucket
   ```
2. **Alternatively, start all services including MinIO:**
   ```bash
   docker compose up -d
   ```

---