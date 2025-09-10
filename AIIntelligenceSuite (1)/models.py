from app import db
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    models = db.relationship('SavedModel', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SavedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(200), nullable=False)
    metrics = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    @property
    def accuracy(self):
        """Return the accuracy from the metrics if it exists"""
        if self.metrics and 'accuracy' in self.metrics:
            return self.metrics['accuracy']
        return None
    
    @property
    def formatted_date(self):
        """Return a formatted date string"""
        return self.created_at.strftime('%B %d, %Y at %H:%M')

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(200), nullable=False)
    row_count = db.Column(db.Integer)
    column_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref='datasets')

class TrainingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    model_id = db.Column(db.Integer, db.ForeignKey('saved_model.id'))
    status = db.Column(db.String(20), default='pending')
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    training_params = db.Column(db.JSON)
    results = db.Column(db.JSON)
    
    dataset = db.relationship('Dataset')
    model = db.relationship('SavedModel')
