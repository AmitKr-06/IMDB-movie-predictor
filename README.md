# IMDB Movie Review Sentiment Predictor

A machine learning pipeline that cleans IMDB movie reviews and predicts sentiment (positive/negative) using classic NLP techniques and classical ML models. The full pipeline is version-controlled and reproducible end-to-end with **DVC**, and every experiment is tracked with **MLflow** on **DagsHub**.

## Overview

This project takes the raw IMDB movie review dataset, cleans and preprocesses the text, engineers features using Bag of Words / TF-IDF, trains and compares several classification models, and registers the best-performing model to the MLflow Model Registry for downstream use.

## Pipeline

The pipeline runs as six sequential DVC stages:

1. **Data Ingestion** — loads the raw dataset and splits it into train/test sets
2. **Data Preprocessing** — cleans review text (lowercasing, stopword removal, punctuation/number/URL stripping, lemmatization)
3. **Feature Engineering** — vectorizes text using Bag of Words (configurable `max_features`)
4. **Model Building** — trains a Logistic Regression classifier
5. **Model Evaluation** — evaluates the model and logs metrics, parameters, and the model artifact to MLflow
6. **Model Registration** — registers the trained model to the MLflow Model Registry and aliases it as `staging`

Every stage is defined in `dvc.yaml` with its dependencies, parameters, and outputs tracked by DVC, so the entire pipeline can be reproduced with a single command.

## Experiments

Before settling on the production pipeline, several experiments were run and tracked in MLflow to compare approaches:

- **Bag of Words vs. TF-IDF** vectorization
- **Five classifiers** compared across both vectorizers: Logistic Regression, Multinomial Naive Bayes, XGBoost, Random Forest, and Gradient Boosting
- **Hyperparameter tuning** for Logistic Regression using `GridSearchCV` (tuning `C`, `penalty`, and `solver`)

All experiment runs, metrics, and parameters are logged and viewable on DagsHub's MLflow tracking UI.

## Current Model Performance

The registered Logistic Regression model achieves the following on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 0.7356 |
| Precision | 0.7310 |
| Recall | 0.7519 |
| AUC | 0.8138 |

## Tech Stack

- **Data handling & ML:** pandas, numpy, scikit-learn, xgboost, lightgbm
- **NLP:** nltk, spacy, gensim
- **Pipeline & versioning:** DVC
- **Experiment tracking:** MLflow, DagsHub
- **Database (optional local experimentation):** SQL Server, pyodbc, SQLAlchemy

## Getting Started

### Prerequisites

- Python 3.12
- A DagsHub account with a token for MLflow tracking

### Setup

1. Clone the repository and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. Create a `.env` file in the project root with your DagsHub credentials:
   ```
   CAPSTONE_TEST=your_dagshub_token
   IMDB_Rating_TEST=your_dagshub_token
   ```

### Running the Pipeline

Reproduce the full pipeline with:
```
dvc repro
```

Check pipeline status at any time with:
```
dvc status
```

## Tracking

Live experiment tracking, metrics, and the model registry are available on DagsHub:

**https://dagshub.com/AmitKr-06/IMDB-movie-predictor.mlflow**

## Configuration

Pipeline parameters (test size, max vectorizer features, etc.) are defined in `params.yaml` and can be adjusted without touching the underlying scripts — DVC will detect the change and re-run only the affected stages on the next `dvc repro`.