from setuptools import setup, find_packages

setup(
    name="movie-recommendation-system",
    version="1.0.0",
    description="MovieLens 100K Recommendation System with XGBoost and Collaborative Filtering",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "joblib>=1.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)