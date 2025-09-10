# PyTorch ML Lab

A comprehensive Machine Learning platform built with Flask, PyTorch, and scikit-learn, providing an interactive interface for model training, evaluation, and predictions.

## 🌟 Features

- **Multiple ML Models**: Support for both traditional ML and deep learning approaches
- **Interactive UI**: Web interface for data upload, model training, and predictions
- **Visualization**: Real-time plots and metrics for model performance
- **Model Management**: Save and load trained models
- **RESTful API**: Endpoints for model inference and management

## 📁 Project Structure

```
PyTorch-ML-Lab/
├── api/                    # API endpoints for Vercel serverless deployment
├── ml_models/             # Machine learning model implementations
│   ├── base_model.py      # Base class for all models
│   ├── deep_learning.py   # PyTorch neural network implementations
│   └── traditional_ml.py  # Scikit-learn based models
├── static/                # Static assets (CSS, JS, images)
├── templates/             # HTML templates
├── utils/                # Utility functions
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   └── visualization.py
├── app.py                # Main Flask application
└── requirements.txt      # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shretadas/PyTorch-ML-Lab.git
   cd PyTorch-ML-Lab
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## 🌐 Deployment Options

### 1. Render Deployment (Recommended for ML)

1. Create a Render account at https://render.com
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Python Version: 3.8 or higher

### 2. AWS Deployment

1. Create an AWS account
2. Set up an EC2 instance or use AWS Elastic Beanstalk
3. Configure environment variables
4. Deploy using AWS CLI or GitHub Actions

### 3. Google Cloud Platform

1. Create a GCP account
2. Enable Cloud Run or App Engine
3. Configure using `app.yaml`
4. Deploy using Cloud SDK

## 💻 API Reference

### Model Training
```http
POST /api/train
Content-Type: application/json

{
    "model_type": "deep_learning",
    "data_path": "path/to/data.csv",
    "parameters": {
        "learning_rate": 0.001,
        "epochs": 100
    }
}
```

### Predictions
```http
POST /api/predict
Content-Type: application/json

{
    "model_id": "model_123",
    "data": [...]
}
```

## 🛠️ Core Components

### 1. ML Models (`ml_models/`)
- `base_model.py`: Abstract base class defining the model interface
- `deep_learning.py`: PyTorch neural network implementations
- `traditional_ml.py`: Scikit-learn based models

### 2. Utils (`utils/`)
- `data_preprocessing.py`: Data cleaning and transformation
- `evaluation.py`: Model evaluation metrics
- `visualization.py`: Plotting and visualization tools

### 3. Web Interface (`templates/`)
- Interactive dashboards for model training
- Visualization of model performance
- Model management interface

## 📊 Visualization Examples

The platform generates various visualizations:
- Learning curves
- Feature importance plots
- Confusion matrices
- Prediction vs Actual plots
- Residual analysis

## 🔐 Environment Variables

Required environment variables:
```
FLASK_SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
MODEL_STORAGE_PATH=path/to/store/models
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Scikit-learn community for machine learning tools
- Flask team for the web framework

## 📧 Contact

For questions and feedback:
- GitHub Issues: [Create an issue](https://github.com/shretadas/PyTorch-ML-Lab/issues)