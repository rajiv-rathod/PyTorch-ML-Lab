from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd
import numpy as np
import joblib
import json
import tempfile

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_ml_project_secret")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/ml_project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["UPLOAD_FOLDER"] = "uploads"
db = SQLAlchemy(app)

@app.route('/')
def index():
    return "Flask API running on Vercel!"

# Add more routes and logic as needed

if __name__ == "__main__":
    app.run()
