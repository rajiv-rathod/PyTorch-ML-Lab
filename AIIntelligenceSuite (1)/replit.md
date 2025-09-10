# PyTorch ML Lab

## Overview

PyTorch ML Lab is an advanced web-based machine learning platform that combines deep learning capabilities using PyTorch with traditional machine learning algorithms from scikit-learn. The application provides a user-friendly interface for data upload, model training, evaluation, and prediction, making machine learning accessible through an intuitive web interface.

The platform supports both classification and regression tasks, offering comprehensive data preprocessing, model evaluation metrics, and visualization capabilities. Users can train neural networks for complex pattern recognition or utilize established algorithms like Random Forest and SVM for various machine learning tasks.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Framework**: Flask-based web application with Jinja2 templating
- **UI Framework**: Bootstrap 5 with dark theme for responsive design
- **Client-side**: Vanilla JavaScript for interactive features like file uploads, feature selection, and model configuration
- **Styling**: Custom CSS with hover effects and responsive layouts

### Backend Architecture
- **Web Framework**: Flask with SQLAlchemy ORM for database operations
- **Model Architecture**: Abstract base class pattern with separate implementations for deep learning and traditional ML
- **Data Processing Pipeline**: Modular preprocessing system with feature encoding, scaling, and missing value handling
- **Session Management**: Flask sessions with configurable secret key
- **File Handling**: Upload system for CSV datasets with temporary file processing

### Machine Learning Components
- **Deep Learning**: PyTorch-based neural networks with customizable architecture, GPU support, and automatic device detection
- **Traditional ML**: Scikit-learn algorithms including Random Forest, Gradient Boosting, SVM, Logistic Regression, and k-NN
- **Preprocessing**: Automated handling of categorical encoding, feature scaling, and data splitting
- **Evaluation**: Comprehensive metrics for both classification and regression tasks
- **Visualization**: Matplotlib and Seaborn for confusion matrices, ROC curves, and feature importance plots

### Database Design
- **ORM**: SQLAlchemy with declarative base pattern
- **Models**: User management, saved model tracking, and dataset storage
- **Relationships**: One-to-many relationships between users and their models/datasets
- **Configuration**: PostgreSQL with connection pooling and health checks

### Authentication & Authorization
- **User Management**: Basic user model with password hashing using Werkzeug
- **Session Handling**: Flask session-based authentication
- **Admin Interface**: Administrative dashboard for user and data management

## External Dependencies

### Core Machine Learning Libraries
- **PyTorch**: Deep learning framework for neural network implementation
- **scikit-learn**: Traditional machine learning algorithms and preprocessing utilities
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations

### Web Framework & Database
- **Flask**: Web application framework with SQLAlchemy integration
- **PostgreSQL**: Primary database for production deployment
- **SQLAlchemy**: ORM for database operations and model definitions

### Data Visualization & Processing
- **matplotlib**: Static plotting and visualization generation
- **seaborn**: Statistical data visualization
- **joblib**: Model serialization and persistence

### Frontend Libraries
- **Bootstrap 5**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements
- **Replit Bootstrap Theme**: Dark theme customization

### Development & Deployment
- **Werkzeug**: WSGI utilities and security helpers
- **ProxyFix**: Middleware for handling reverse proxy headers
- **tempfile**: Temporary file handling for data processing