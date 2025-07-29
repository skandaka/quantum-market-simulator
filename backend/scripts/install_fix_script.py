"""
Ultimate fix script for Quantum Market Simulator
"""

import subprocess
import sys
import os
import shutil


def run_command(cmd, check=True):
    """Run command and return success status"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return False


def fix_pydantic_conflicts():
    """Fix pydantic version conflicts"""
    print("\\n🔧 Fixing pydantic conflicts...")

    # Uninstall conflicting packages
    packages_to_remove = [
        "pydantic", "pydantic-core", "pydantic-settings",
        "fastapi", "classiq"
    ]

    for pkg in packages_to_remove:
        run_command(f"{sys.executable} -m pip uninstall -y {pkg}", check=False)

    # Install compatible versions in correct order
    install_commands = [
        f"{sys.executable} -m pip install 'pydantic>=2.4.0,<2.10.0'",
        f"{sys.executable} -m pip install 'pydantic-settings>=2.4.0,<3.0.0'",
        f"{sys.executable} -m pip install 'fastapi==0.104.1'",
    ]

    for cmd in install_commands:
        if not run_command(cmd):
            print(f"Failed: {cmd}")
            return False

    return True


def install_core_deps():
    """Install core dependencies"""
    print("\\n📦 Installing core dependencies...")

    core_deps = [
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0",
        "numpy==1.26.2",
        "pandas==2.1.4",
        "scipy==1.11.4",
    ]

    for dep in core_deps:
        if not run_command(f"{sys.executable} -m pip install '{dep}'"):
            print(f"Warning: Failed to install {dep}")


def install_ml_deps():
    """Install ML dependencies with fallbacks"""
    print("\\n🧠 Installing ML dependencies...")

    # Try to install transformers without downloading models
    run_command(f"{sys.executable} -m pip install 'transformers==4.35.2' --no-deps")
    run_command(f"{sys.executable} -m pip install 'torch>=2.2.0' --index-url https://download.pytorch.org/whl/cpu")
    run_command(f"{sys.executable} -m pip install 'scikit-learn==1.3.2'")
    run_command(f"{sys.executable} -m pip install 'textblob==0.17.1'")


def fix_news_processor():
    """Fix news_processor.py"""
    print("\\n📰 Fixing news processor...")

    news_processor_path = "app/services/news_processor.py"
    if os.path.exists(news_processor_path):
        # Create backup
        shutil.copy(news_processor_path, news_processor_path + ".backup")

        # Read and fix content
        with open(news_processor_path, 'r') as f:
            content = f.read()

        # Remove newspaper imports
        content = content.replace("from newspaper import Article", "# from newspaper import Article")
        content = content.replace("import newspaper", "# import newspaper")

        # Write fixed content
        with open(news_processor_path, 'w') as f:
            f.write(content)

        print("✅ News processor fixed")


def create_models_cache_dir():
    """Create models cache directory"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print("✅ Created cache directories")


def final_install():
    """Final installation from requirements.txt"""
    print("\\n🎯 Final installation...")

    if os.path.exists("requirements.txt"):
        # Install remaining packages, ignoring conflicts
        run_command(f"{sys.executable} -m pip install -r requirements.txt --no-deps", check=False)
        run_command(f"{sys.executable} -m pip install -r requirements.txt", check=False)


def main():
    """Main fix function"""
    print("🚀 Ultimate Quantum Market Simulator Fix")
    print("=" * 60)

    # Step 1: Update pip
    print("\\n1. Updating pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Step 2: Fix pydantic conflicts
    if not fix_pydantic_conflicts():
        print("❌ Failed to fix pydantic conflicts")
        return False

    # Step 3: Install core dependencies
    install_core_deps()

    # Step 4: Install ML dependencies
    install_ml_deps()

    # Step 5: Fix news processor
    fix_news_processor()

    # Step 6: Create directories
    create_models_cache_dir()

    # Step 7: Final installation
    final_install()

    # Step 8: Try to download spacy model
    print("\\n🔤 Downloading spaCy model...")
    run_command(f"{sys.executable} -m spacy download en_core_web_sm", check=False)

    print("\\n" + "=" * 60)
    print("✅ Fix complete!")
    print("\\nTo run the application:")
    print("1. cd backend")
    print("2. source venv/bin/activate  # On Windows: venv\\\\Scripts\\\\activate")
    print("3. python -m app.main")

    return True


if __name__ == "__main__":
    main()