# Makefile for automating the equity-aware tutoring pipeline
# Author: Amit Pimpalkar, RBU Nagpur

.PHONY: help install data-prep train evaluate distill tune serve test clean

help:
	@echo "Available commands:"
	@echo "  install      - Install Python dependencies"
	@echo "  data-prep    - Download and preprocess datasets"
	@echo "  train        - Train the teacher model"
	@echo "  evaluate     - Evaluate the trained model"
	@echo "  distill      - Compress teacher into student policy"
	@echo "  tune         - Run hyperparameter tuning"
	@echo "  serve        - Start FastAPI server"
	@echo "  test         - Run unit tests"
	@echo "  clean        - Remove temporary files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

data-prep:
	python scripts/download_data.py --dataset all
	python scripts/prepare_data.py --dataset ednet
	python scripts/prepare_data.py --dataset assistments
	python scripts/prepare_data.py --dataset oulad

train-ednet:
	python scripts/train_model.py --dataset ednet --config configs/training_config.yaml

train-assistments:
	python scripts/train_model.py --dataset assistments --config configs/training_config.yaml

train-oulad:
	python scripts/train_model.py --dataset oulad --config configs/training_config.yaml

train: train-ednet train-assistments train-oulad

evaluate:
	python scripts/evaluate_model.py --checkpoint outputs/experiments/*/checkpoints/best_model.pth --dataset ednet

distill:
	python scripts/distill_model.py --teacher outputs/experiments/*/checkpoints/best_model.pth --dataset ednet

tune:
	python scripts/hyperparameter_tuning.py --dataset ednet --trials 20

serve:
	python src/serving/app.py

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf outputs/experiments/*/logs
	rm -rf outputs/figures/*.png