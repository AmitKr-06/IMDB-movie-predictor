import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer
import scipy.sparse
from nltk.corpus import stopwords

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

#============================ CONFIGURATION ====================================
CONFIG = {
    "data_path": "notebooks/IMDB.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/AmitKr-06/IMDB-movie-predictor.mlflow",
    "dagshub_repo_owner": "AmitKr-06",
    "dagshub_repo_name": "IMDB-movie-predictor",
    "experiment_name": "BOW vs TFIDF"
}


# ============================== SETUP MLFLOW & DAGSHUB ===========================================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])


# ============================= TEXT PREPROCESSING ========================================================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ================================== LOAD & PREPROCESS DATA ========================================

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ================================== FEATURE ENGINEERING ===========================================

VECTORIZERS = {
    'BOW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBClassifier': XGBClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}


# ================================= SKOPS TRUSTED TYPES ====================================================

def get_trusted_types(algo_name):
    """Return skops trusted types needed to safely load each model type."""
    if algo_name in ['RandomForestClassifier', 'GradientBoostingClassifier']:
        return ["sklearn.tree._tree.Tree"]
    elif algo_name == 'XGBClassifier':
        return ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
    else:
        return []


# ================================= TRAIN & EVALUATE MODELS ====================================================

def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        # Feature extraction
                        X = vectorizer.fit_transform(df['review'])
                        y = df['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

                        # Log Preprocessing Parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # Train model
                        model = algorithm
                        model.fit(X_train, y_train)

                        # Log model parameters
                        log_model_params(algo_name, model)

                        # Evaluate model
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred)
                        }
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()

                        # Only pass skops_trusted_types when there's something to trust,
                        # since this mlflow version's save_model() doesn't accept an empty/unknown kwarg cleanly
                        trusted_types = get_trusted_types(algo_name)
                        log_model_kwargs = {"input_example": input_example}
                        if trusted_types:
                            log_model_kwargs["skops_trusted_types"] = trusted_types

                        mlflow.sklearn.log_model(model, "model", **log_model_kwargs)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))


def log_model_params(algo_name, model):
    """ Logs hyperparameters of the trained model to MLFLOW."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBClassifier':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForestClassifier':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoostingClassifier':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)


# ==================================== EXECUTION ========================================================

if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)