# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 10GB disk space

### Recommended Requirements
- Python 3.8 or higher
- 8GB+ RAM
- 20GB+ disk space
- NVIDIA GPU (for deep learning)

## Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shretadas/PyTorch-ML-Lab.git
cd PyTorch-ML-Lab
```

### 2. Set Up Python Environment

#### Using venv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Using conda
```bash
conda create -n pytorch-ml-lab python=3.8
conda activate pytorch-ml-lab
```

### 3. Install Dependencies

#### Basic Installation
```bash
pip install -r requirements.txt
```

#### With GPU Support
```bash
pip install -r requirements-gpu.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:
```bash
FLASK_SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///instance/ml_project.db
MODEL_STORAGE_PATH=uploads/models
```

### 5. Initialize Database
```bash
python
>>> from app import db
>>> db.create_all()
>>> exit()
```

### 6. Run the Application
```bash
python app.py
```

## Verification

1. Open web browser to `http://localhost:5000`
2. Verify the following pages load:
   - Home page
   - Model training page
   - Data upload page
   - Visualization dashboard

## Troubleshooting

### Common Issues

#### 1. Package Installation Errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

#### 2. Database Initialization Errors
```bash
rm -rf instance/ml_project.db
python
>>> from app import db
>>> db.create_all()
>>> exit()
```

#### 3. GPU Issues
- Verify CUDA installation
- Check PyTorch GPU support
- Update GPU drivers

## Next Steps

After installation:
1. Upload a sample dataset
2. Train your first model
3. Explore the visualization tools
4. Read the API documentation
