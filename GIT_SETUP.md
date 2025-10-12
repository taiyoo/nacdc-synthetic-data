# Git Repository Setup Guide

## 🔧 Setting Up Git Repository

### 1. Initialize Git Repository
```bash
cd /path/to/your/project
git init
git add .
git commit -m "Initial commit: NACDC Synthetic Data Generation with Differential Privacy"
```

### 2. Connect to Remote Repository
```bash
# Add your remote repository URL
git remote add origin https://github.com/taiyoo/nacdc-synthetic-data.git

# Push to remote
git branch -M main
git push -u origin main
```

### 3. For New Users Cloning the Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/nacdc-synthetic-data.git
cd nacdc-synthetic-data

# Run automated setup
python setup.py

# Or manual setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🎯 Testing Path Resolution

The project now works correctly from any directory:

```python
# Test dynamic paths work correctly
from src.government_compliance_analyzer import GovernmentCompliantAnalyzer

analyzer = GovernmentCompliantAnalyzer()
print(f"Data directory: {analyzer.data_dir}")
# Output: /path/to/project/data/government_compliant
```

## 📂 .gitignore Configuration

The `.gitignore` file excludes:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`, `venv/`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Excel temp files (`~$*.xlsx`)

**Note**: CSV data files are **included** by default. Uncomment lines in `.gitignore` if you want to exclude large datasets.

## ✅ Ready for Distribution

## 🚀 Quick Commands for Git Users

```bash
# Clone and setup
git clone <your-repo-url>
cd <project-name>
python setup.py

# Generate government-compliant data
source .venv/bin/activate
python src/government_compliant_generator.py

# Analyze compliance
python src/government_compliance_analyzer.py

# Generate research data
python src/nacdc_synthetic_dp.py
```
