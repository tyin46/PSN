"""
BART Evaluation Package Setup
Checks dependencies and provides installation guidance
"""

import sys
import subprocess
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Check and install required packages for BART evaluation"""
    
    print("BART Evaluation Setup")
    print("="*50)
    
    # Core packages needed for evaluation (not GUI)
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"), 
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    # Optional packages
    optional_packages = [
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
        ("customtkinter", "customtkinter"),
        ("langchain-community", "langchain_community"),
    ]
    
    print("\nChecking required packages...")
    missing_required = []
    
    for package, import_name in required_packages:
        if check_package(package, import_name):
            print(f"✓ {package}")
        else:
            print(f"✗ {package} (required)")
            missing_required.append(package)
    
    print("\nChecking optional packages...")
    missing_optional = []
    
    for package, import_name in optional_packages:
        if check_package(package, import_name):
            print(f"✓ {package}")
        else:
            print(f"- {package} (optional)")
            missing_optional.append(package)
    
    # Install missing required packages
    if missing_required:
        print(f"\nInstalling {len(missing_required)} required packages...")
        for package in missing_required:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
                print(f"Please run: pip install {package}")
    
    # Provide guidance for optional packages
    if missing_optional:
        print(f"\nOptional packages not found:")
        for package in missing_optional:
            print(f"- {package}: pip install {package}")
        
        print(f"\nNote: Optional packages enable additional features:")
        print(f"- openai: Real LLM evaluation with GPT models")
        print(f"- python-dotenv: Load API keys from .env file")
        print(f"- customtkinter: Modern GUI for interactive testing")
        print(f"- langchain-community: Integration with local LLM models")
    
    print(f"\nSetup complete!")
    
    # Check what evaluations can be run
    print(f"\nAvailable evaluations:")
    print(f"✓ quick_bart_evaluation.py - Fast simulation (no external deps)")
    
    if check_package("openai"):
        print(f"✓ bart_persona_evaluation.py - Real LLM evaluation")
    else:
        print(f"- bart_persona_evaluation.py - Requires: pip install openai python-dotenv")
    
    print(f"✓ bart_analysis.py - Results analysis and visualization")
    
    if check_package("customtkinter"):
        print(f"✓ Interactive GUI testing available")
    else:
        print(f"- Interactive GUI - Requires: pip install customtkinter")
    
    print(f"\nTo get started:")
    print(f"1. Run: python quick_bart_evaluation.py")
    print(f"2. Run: python bart_analysis.py")
    print(f"3. Check generated images and results!")

if __name__ == "__main__":
    main()