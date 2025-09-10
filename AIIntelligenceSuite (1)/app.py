import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
import pandas as pd
import numpy as np
import joblib
import json
import tempfile

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_ml_project_secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# configure the database using PostgreSQL
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
import os
db_path = os.path.join(os.path.dirname(__file__), 'instance', 'ml_project.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["UPLOAD_FOLDER"] = "uploads"

# Initialize the app with the extension
db.init_app(app)

# Import routes after app is initialized to avoid circular imports
from ml_models.deep_learning import DeepLearningModel
from ml_models.traditional_ml import TraditionalMLModel
from utils.data_preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and file.filename and file.filename.endswith('.csv'):
        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            file.save(temp_file.name)
            session['dataset_path'] = temp_file.name
            
        try:
            # Preview the dataset
            df = pd.read_csv(session['dataset_path'])
            preview = df.head(5).to_html(classes='table table-striped table-sm')
            columns = df.columns.tolist()
            
            # Store in session
            session['dataset_columns'] = columns
            
            return render_template('index.html', 
                                   preview=preview, 
                                   columns=columns,
                                   dataset_loaded=True)
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    if 'dataset_path' not in session:
        flash('Please upload a dataset first')
        return redirect(url_for('index'))
    
    try:
        # Get form data
        target_column = request.form.get('target_column')
        model_type = request.form.get('model_type')
        feature_columns = request.form.getlist('feature_columns')
        test_size_str = request.form.get('test_size', '0.2')
        if test_size_str.lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
            flash('Invalid test size value. Please enter a valid number.')
            return redirect(url_for('index'))
        test_size = float(test_size_str)
        
        if not target_column or not model_type or not feature_columns:
            flash('Please select target column, features, and model type')
            return redirect(url_for('index'))
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        df = pd.read_csv(session['dataset_path'])
        
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
            df, feature_columns, target_column, test_size=test_size
        )
        
        # Train model based on selection
        if model_type == 'deep_learning':
            model = DeepLearningModel(
                input_dim=len(feature_columns),
                hidden_dim=int(request.form.get('hidden_dim', '64')) if request.form.get('hidden_dim', '64').lower() not in ['nan', 'inf', '-inf', 'infinity', '-infinity'] else 64,
                output_dim=1 if len(np.unique(y_train)) == 2 else len(np.unique(y_train)),
                learning_rate=float(request.form.get('learning_rate', '0.001')) if request.form.get('learning_rate', '0.001').lower() not in ['nan', 'inf', '-inf', 'infinity', '-infinity'] else 0.001,
                epochs=int(request.form.get('epochs', '100')) if request.form.get('epochs', '100').lower() not in ['nan', 'inf', '-inf', 'infinity', '-infinity'] else 100
            )
            is_classification = len(np.unique(y_train)) < 10  # Heuristic
            model_info = model.train(X_train, y_train, X_test, y_test, is_classification=is_classification)
        else:
            model = TraditionalMLModel(
                model_name=request.form.get('traditional_model', 'random_forest'),
                params={}
            )
            is_classification = len(np.unique(y_train)) < 10  # Heuristic
            model_info = model.train(X_train, y_train, X_test, y_test, is_classification=is_classification)
        
        # Evaluate model
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test, is_classification=is_classification)
        
        # Create visualizations
        visualizer = Visualizer()
        plot_paths = visualizer.create_visualizations(model, X_test, y_test, is_classification=is_classification)
        
        # Save model for later prediction
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl')
        feature_path = os.path.join(app.config['UPLOAD_FOLDER'], 'features.json')
        
        joblib.dump(model, model_path)
        with open(feature_path, 'w') as f:
            json.dump({
                'features': feature_columns,
                'target': target_column,
                'is_classification': is_classification
            }, f)
        
        # Store training results in session
        session['training_results'] = {
            'metrics': metrics,
            'plots': plot_paths,
            'model_type': model_type,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'is_classification': is_classification
        }
        
        return render_template('results.html', 
                              metrics=metrics, 
                              plots=plot_paths,
                              model_type=model_type,
                              training_info=model_info)
    
    except Exception as e:
        logger.exception("Error during model training")
        flash(f'Error training model: {str(e)}')
        return redirect(url_for('index'))

@app.route('/predict')
def predict_page():
    if 'training_results' not in session:
        flash('Please train a model first')
        return redirect(url_for('index'))
    
    feature_columns = session['training_results']['feature_columns']
    return render_template('predict.html', features=feature_columns)

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    if 'training_results' not in session:
        flash('Please train a model first')
        return redirect(url_for('index'))
    
    try:
        # Load saved model and features
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl')
        feature_path = os.path.join(app.config['UPLOAD_FOLDER'], 'features.json')
        
        model = joblib.load(model_path)
        with open(feature_path, 'r') as f:
            feature_info = json.load(f)
        
        # Get input values from form
        input_data = {}
        for feature in feature_info['features']:
            value_str = str(request.form.get(feature, '0'))
            if value_str.lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
                flash(f'Invalid value for {feature}. Please enter a valid number.')
                return redirect(url_for('predict_page'))
            input_data[feature] = float(value_str)
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        result = model.predict(input_df)
        
        if feature_info['is_classification']:
            prediction = int(result[0])
            return render_template('predict.html', 
                                 features=feature_info['features'],
                                 prediction=prediction,
                                 prediction_made=True)
        else:
            prediction = float(result[0])
            return render_template('predict.html', 
                                 features=feature_info['features'],
                                 prediction=prediction,
                                 prediction_made=True)
    
    except Exception as e:
        logger.exception("Error during prediction")
        flash(f'Error making prediction: {str(e)}')
        return redirect(url_for('predict_page'))

# Admin routes
@app.route('/admin')
def admin_dashboard():
    users = models.User.query.all()
    saved_models = models.SavedModel.query.all()
    datasets = models.Dataset.query.all()
    return render_template('admin.html', users=users, models=saved_models, datasets=datasets)

@app.route('/admin/add_user', methods=['POST'])
def add_user():
    try:
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = models.User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        # Create new user
        user = models.User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('User added successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logger.exception("Error adding user")
        flash(f'Error adding user: {str(e)}', 'danger')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    user = models.User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        try:
            user.username = request.form.get('username')
            user.email = request.form.get('email')
            
            # Only update password if provided
            password = request.form.get('password')
            if password:
                user.set_password(password)
            
            db.session.commit()
            flash('User updated successfully', 'success')
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            db.session.rollback()
            logger.exception("Error updating user")
            flash(f'Error updating user: {str(e)}', 'danger')
    
    return render_template('edit_user.html', user=user)

@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    try:
        user = models.User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logger.exception("Error deleting user")
        flash(f'Error deleting user: {str(e)}', 'danger')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/view_model/<int:model_id>')
def view_model(model_id):
    model = models.SavedModel.query.get_or_404(model_id)
    return render_template('view_model.html', model=model)

@app.route('/admin/delete_model/<int:model_id>')
def delete_model(model_id):
    try:
        model = models.SavedModel.query.get_or_404(model_id)
        
        # Delete the model file if it exists
        if os.path.exists(model.file_path):
            os.remove(model.file_path)
        
        db.session.delete(model)
        db.session.commit()
        flash('Model deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logger.exception("Error deleting model")
        flash(f'Error deleting model: {str(e)}', 'danger')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/view_dataset/<int:dataset_id>')
def view_dataset(dataset_id):
    dataset = models.Dataset.query.get_or_404(dataset_id)
    
    # Read a preview of the dataset if the file exists
    preview = None
    if os.path.exists(dataset.file_path):
        try:
            df = pd.read_csv(dataset.file_path)
            preview = df.head(10).to_html(classes='table table-dark table-striped')
        except Exception as e:
            logger.exception("Error reading dataset")
            flash(f'Error reading dataset: {str(e)}', 'warning')
    
    return render_template('view_dataset.html', dataset=dataset, preview=preview)

@app.route('/admin/delete_dataset/<int:dataset_id>')
def delete_dataset(dataset_id):
    try:
        dataset = models.Dataset.query.get_or_404(dataset_id)
        
        # Delete the dataset file if it exists
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        db.session.delete(dataset)
        db.session.commit()
        flash('Dataset deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logger.exception("Error deleting dataset")
        flash(f'Error deleting dataset: {str(e)}', 'danger')
    
    return redirect(url_for('admin_dashboard'))


with app.app_context():
    # Import the models here so their tables will be created
    import models  # noqa: F401
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
