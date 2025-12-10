# 🚀 Quick Start Guide

## Deploy to GitHub & Streamlit in 5 Minutes!

### Step 1: Push to GitHub (2 minutes)

```bash
# Navigate to project
cd /Users/abc/Downloads/ResourceRL

# Add all files
git add .

# Commit
git commit -m "Add ResourceRL with Streamlit app"

# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/ResourceRL.git
git branch -M main
git push -u origin main
```

**Don't have a GitHub repo yet?**
1. Go to https://github.com/new
2. Name it: `ResourceRL`
3. Don't initialize with README
4. Copy the repository URL
5. Use it in the `git remote add` command above

### Step 2: Deploy to Streamlit Cloud (3 minutes)

1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/ResourceRL`
5. Main file: `app.py`
6. Click **"Deploy"**
7. Wait 1-2 minutes
8. **Done!** Your app is live! 🎉

## 🧪 Test Locally First (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Visit: http://localhost:8501

## 📱 What You'll Get

A beautiful web app with:
- 🎓 **Train Agent**: Train RL models through UI
- 📊 **Evaluate Model**: Test performance
- 🎮 **Live Demo**: Watch agent make decisions
- 📈 **Visualizations**: Charts and metrics

## ⚠️ Important

- Models are large and in `.gitignore` (won't upload)
- App will train models on-demand
- First training may take 1-2 minutes
- Use 10k-30k timesteps for faster training

## 🆘 Need Help?

- See `GITHUB_SETUP.md` for detailed instructions
- See `DEPLOYMENT.md` for deployment options
- Check Streamlit Cloud logs if deployment fails

## ✅ Files Ready for GitHub

All these files are ready to commit:
- ✅ `app.py` - Streamlit web app
- ✅ `resource_scheduler_env.py` - RL environment
- ✅ `train_agent.py` - Training script
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules
- ✅ `.streamlit/config.toml` - Streamlit config
- ✅ All documentation files

**Ready to deploy!** 🚀

