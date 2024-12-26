# Installation
```bash
apt-get update && apt-get install -y --no-install-recommends \
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
```

```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Create a ROI and Image Template

```bash
python3 main.py --setting
```

## Run Defect Detection (Milk Carton Metallic Detection and MFD/EXD Text Extraction)

```bash
python3 main.py --main_disp --process_disp --debug
```

- **--main_disp** : Show main GUI
- **--process_disp** : Show process GUI
- **--debug**: Show debug log

# Docker
```bash
docker compose up -d --build
```

## Run defect-detection
```bash
docker compose up -d fn-defect-detection
```

## (Optional) Save Alert Image to Minio 
```bash
docker compose up -d 
```
or

```bash
dockr compose up -d minio minio-create-bucket
```
