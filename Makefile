# Makefile for Movie Recommendation System

.PHONY: install data train test demo api clean

# Install dependencies
install:
	pip install -r requirements.txt

# Prepare data
data:
	python scripts/create_data.py

# Train models
train:
	python scripts/train_model.py

# Run cross-validation
validate:
	python scripts/cross_validation.py

# Run tests
test:
	python -m pytest tests/ -v

# Run demo
demo:
	python scripts/demo.py

# Start API server
api:
	python api.py

# Clean generated files
clean:
	rm -rf models/*.pkl
	rm -rf data/ml100k_combined.csv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Full pipeline
all: install data train validate

# Development setup
dev-setup: install
	pip install pytest jupyter matplotlib seaborn

# Run feature inspection
inspect:
	python scripts/inspect_features.py