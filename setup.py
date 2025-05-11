from setuptools import setup, find_packages

setup(
    name="scouting_ocr_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "python-dotenv>=1.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "azure-ai-formrecognizer>=3.2.0",
        "azure-core>=1.29.0",
        "azure-common>=1.1.28"
    ]
) 