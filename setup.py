#!/usr/bin/env python3
"""
Setup script for NACDC Synthetic Data Generation Project
Installs dependencies and creates necessary directories
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and print the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False
    return True

def main():
    print("🏛️  NACDC Synthetic Data Generation Project Setup")
    print("=" * 60)
    
    # Get project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"📂 Project directory: {project_root}")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    # Create virtual environment if it doesn't exist
    venv_path = project_root / ".venv"
    if not venv_path.exists():
        if not run_command("python -m venv .venv", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate &&"
        python_cmd = ".venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        activate_cmd = "source .venv/bin/activate &&"
        python_cmd = ".venv/bin/python"
    
    # Install requirements
    print("📦 Installing Python packages...")
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "diffprivlib>=0.6.0",
        "faker>=20.0.0",
        "openpyxl>=3.0.0"
    ]
    
    for package in packages:
        if not run_command(f"{activate_cmd} pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Create necessary directories
    directories = [
        "data/synthetic",
        "data/government_compliant", 
        "data/specifications",
        "analysis",
        "unused"
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Generate government-compliant data:")
    print(f"   {python_cmd} src/government_compliant_generator.py")
    print("\n2. Generate research data:")
    print(f"   {python_cmd} src/nacdc_synthetic_dp.py")
    print("\n3. Analyze datasets:")
    print(f"   {python_cmd} src/government_compliance_analyzer.py")
    print(f"   {python_cmd} src/nacdc_analysis.py")
    
    print("\n📖 Documentation:")
    print("   - Main README: README.md")
    print("   - Government data: data/government_compliant/README.md")
    print("   - Research data: data/README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
