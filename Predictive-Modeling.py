# predictive_modeling.py

"""
Predictive Modeling Module for Personalized Healthcare Recommendation System

This module contains functions for building, training, and evaluating predictive models
to forecast patient health outcomes and recommend personalized interventions.

Techniques Used:
- Regression
- Classification
- Clustering
- Ensemble methods

Algorithms Used:
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- K-Means Clustering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class PredictiveModeling:
    def __init__(self):
        """
        Initialize the PredictiveModeling class.
        """
        self.models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeClassifier(),
            'random_forest': RandomForestClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'kmeans': KMeans(n_clusters=3)
        }

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath)

    def split_data(self, data, target_column):
        """
        Split the data into training and testing sets.
        
        :param data: DataFrame, input data
        :param target_column: str, name of the target column
        :return: tuples, training and testing sets (X_train, X_test, y_train, y_test)
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, model_name, X_train, y_train):
        """
        Train the specified model.
        
        :param model_name: str, name of the model to train
        :param X_train: DataFrame, training features
        :param y_train: Series, training target
        """
        model = self.models.get(model_name)
        if model:
            model.fit(X_train, y_train)
            return model
        else:
            raise ValueError(f"Model {model_name} not found.")

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the specified model.
        
        :param model: trained model
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        }
        return metrics

if __name__ == "__main__":
    filepath = 'data/processed/preprocessed_patient_data.csv'
    target_column = 'target'

    predictive_modeling = PredictiveModeling()
    data = predictive_modeling.load_data(filepath)
    X_train, X_test, y_train, y_test = predictive_modeling.split_data(data, target_column)

    model_name = 'random_forest'
    model = predictive_modeling.train_model(model_name, X_train, y_train)
    metrics = predictive_modeling.evaluate_model(model, X_test, y_test)

    print(f"Evaluation Metrics for {model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Save the trained model
    import joblib
    joblib.dump(model, f'models/{model_name}_model.pkl')
    print(f"Trained model saved as models/{model_name}_model.pkl")
