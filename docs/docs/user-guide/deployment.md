# Deployment Guide

## Deployment Options

This guide covers the deployment options for PyTorch ML Lab, with detailed instructions for each platform.

## 1. Render Deployment (Recommended)

Render is recommended for its ease of use and good support for Python ML applications.

### Setup Steps

1. **Create a Render Account**
   - Go to [render.com](https://render.com)
   - Sign up for a new account

2. **Connect Your Repository**
   - Click "New +"
   - Select "Web Service"
   - Connect your GitHub repository

3. **Configure the Service**
   ```yaml
   Name: pytorch-ml-lab
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   ```

4. **Set Environment Variables**
   ```
   FLASK_SECRET_KEY=your_secret_key
   DATABASE_URL=your_database_url
   MODEL_STORAGE_PATH=/var/lib/models
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for the build to complete

## 2. AWS Deployment

### Option A: AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize EB**
   ```bash
   eb init -p python-3.8 pytorch-ml-lab
   ```

3. **Configure Environment**
   Create `.ebextensions/01_flask.config`:
   ```yaml
   option_settings:
     aws:elasticbeanstalk:container:python:
       WSGIPath: app:app
   ```

4. **Deploy**
   ```bash
   eb create pytorch-ml-lab-env
   ```

### Option B: EC2 Instance

1. **Launch EC2 Instance**
   - Choose Ubuntu Server 20.04 LTS
   - Select instance type (t2.large recommended)
   - Configure security groups

2. **SSH into Instance**
   ```bash
   ssh -i key.pem ubuntu@your-instance-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip nginx
   ```

4. **Clone and Setup**
   ```bash
   git clone https://github.com/shretadas/PyTorch-ML-Lab.git
   cd PyTorch-ML-Lab
   pip3 install -r requirements.txt
   ```

5. **Configure Nginx**
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## 3. Google Cloud Platform

### Cloud Run Deployment

1. **Install Google Cloud SDK**
   - Download from [cloud.google.com/sdk](https://cloud.google.com/sdk)

2. **Initialize Project**
   ```bash
   gcloud init
   ```

3. **Build Container**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/pytorch-ml-lab
   ```

4. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy pytorch-ml-lab \
     --image gcr.io/PROJECT_ID/pytorch-ml-lab \
     --platform managed
   ```

### App Engine Deployment

1. **Create app.yaml**
   ```yaml
   runtime: python38
   
   instance_class: F2
   
   env_variables:
     FLASK_SECRET_KEY: "your-secret-key"
   
   entrypoint: gunicorn -b :$PORT app:app
   ```

2. **Deploy**
   ```bash
   gcloud app deploy
   ```

## Environment Variables

### Required Variables
```
FLASK_SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
MODEL_STORAGE_PATH=path/to/store/models
```

### Optional Variables
```
DEBUG=True/False
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## Monitoring and Maintenance

### Health Checks

Implement a health check endpoint:
```python
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})
```

### Logging

Configure logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Backup Strategy

1. **Database Backups**
   - Schedule regular backups
   - Store in secure location
   - Test restoration process

2. **Model Backups**
   - Version control trained models
   - Store in object storage
   - Document model versions

## Security Considerations

1. **SSL/TLS Configuration**
   - Enable HTTPS
   - Configure SSL certificates
   - Redirect HTTP to HTTPS

2. **Access Control**
   - Implement authentication
   - Set up API keys
   - Rate limiting

3. **Data Protection**
   - Encrypt sensitive data
   - Secure file uploads
   - Regular security audits

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Check instance size
   - Monitor memory usage
   - Implement caching

2. **Performance Problems**
   - Profile application
   - Optimize database queries
   - Cache heavy computations

3. **Deployment Failures**
   - Check logs
   - Verify dependencies
   - Test locally first
