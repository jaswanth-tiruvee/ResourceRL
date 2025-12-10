# Deployment Guide

## 🚀 Deploying to Streamlit Cloud

### Step 1: Push to GitHub

1. **Initialize Git repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ResourceRL app"
   ```

2. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name it `ResourceRL` (or your preferred name)
   - Don't initialize with README (we already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/ResourceRL.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure your app**:
   - **Repository**: Select `YOUR_USERNAME/ResourceRL`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (optional)

5. **Click "Deploy"**

6. **Wait for deployment** (usually 1-2 minutes)

7. **Your app is live!** 🎉

### Important Notes for Streamlit Cloud

- **Model Files**: The app will train models on-demand. For faster performance, you can:
  - Pre-train models and commit them (if small)
  - Use Streamlit's file caching
  - Train models in the background

- **Resource Limits**: 
  - Free tier has memory/CPU limits
  - Training large models may timeout
  - Consider using smaller timesteps for training

- **Environment Variables**: Not needed for this app, but you can add them in Streamlit Cloud settings if needed

## 📦 Local Deployment

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🔧 Troubleshooting

### Common Issues

1. **Model not found error**:
   - Train a model first using the "Train Agent" page
   - Or download a pre-trained model

2. **Memory errors during training**:
   - Reduce training timesteps
   - Use A2C instead of PPO (lighter)
   - Train in smaller batches

3. **Slow performance**:
   - This is normal for RL training
   - Consider training models offline and uploading them
   - Use smaller models for demo purposes

## 🌐 Alternative Deployment Options

### Heroku

1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy using Heroku CLI

### Docker

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run:
   ```bash
   docker build -t resourcerl .
   docker run -p 8501:8501 resourcerl
   ```

### AWS/GCP/Azure

- Use container services (ECS, Cloud Run, Container Instances)
- Deploy using the Dockerfile above
- Configure auto-scaling as needed

