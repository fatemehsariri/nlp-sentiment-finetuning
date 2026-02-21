# Multiclass Sentiment Analysis using Transformer Fine-Tuning

## Overview
This project presents a multiclass sentiment classification system based on a pretrained Transformer model. The model is fine-tuned on the Twitter Airline Sentiment dataset and evaluated using standard performance metrics. The workflow includes data preprocessing, tokenization, model training, evaluation, and visualization of results.

## Dataset
The dataset used in this project is the Twitter Airline Sentiment dataset, consisting of three classes:
- Negative
- Neutral
- Positive

A stratified split was applied to create training, validation, and test sets to ensure balanced class distribution.

## Model
The project utilizes a pretrained Transformer model:
- `cardiffnlp/twitter-roberta-base-sentiment`

The model was fine-tuned for a three-class classification task using the HuggingFace Transformers framework.

## Training Configuration
- Framework: HuggingFace Transformers
- Optimizer: AdamW (default in Trainer)
- Learning Rate: 5e-6
- Batch Size: 16
- Number of Epochs: 3
- Weight Decay: 0.01
- Evaluation Strategy: Step-based evaluation
- Best model selected based on F1-score

## Evaluation Metrics
The model was evaluated using:
- Accuracy
- Weighted F1-score
- Classification Report
- Confusion Matrix

## Visualizations
The project includes:
- Training loss curve
- Validation loss curve
- Validation accuracy curve
- Confusion matrix heatmap

These visualizations help analyze model convergence and performance behavior during training.

## Results
The fine-tuned Transformer model achieves strong performance on the test set, demonstrating the effectiveness of deep learning approaches for sentiment classification.

## Technologies Used
- Python
- PyTorch
- HuggingFace Transformers
- Datasets
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## How to Run

Install dependencies:

pip install transformers torch datasets scikit-learn matplotlib seaborn kagglehub

Then run the notebook in a Python environment with GPU support (recommended).

## Project Structure

project/
├── notebook.ipynb

├── images/
│   ├── training_validation_curves.png
│   ├── confusion_matrix.png

└── README.md

## Future Improvements
- Experiment with different hyperparameters
- Apply class weighting to handle imbalance
- Implement early stopping
- Compare with classical machine learning baselines
