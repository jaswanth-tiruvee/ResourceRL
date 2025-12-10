#!/bin/bash
# Script to help set up GitHub repository

echo "🚀 ResourceRL - GitHub Setup"
echo "============================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
fi

# Add all files
echo "Adding files to git..."
git add .

# Check if remote exists
if git remote | grep -q "origin"; then
    echo ""
    echo "⚠️  Remote 'origin' already exists."
    echo "Current remote URL:"
    git remote get-url origin
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your GitHub repository URL: " repo_url
        git remote set-url origin "$repo_url"
    fi
else
    echo ""
    echo "📝 To connect to GitHub:"
    echo "1. Create a new repository on GitHub (https://github.com/new)"
    echo "2. Then run:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/ResourceRL.git"
    echo "   git push -u origin main"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Commit changes: git commit -m 'Add Streamlit app and deployment files'"
echo "3. Push to GitHub: git push -u origin main"
echo "4. Deploy to Streamlit Cloud: https://share.streamlit.io/"

