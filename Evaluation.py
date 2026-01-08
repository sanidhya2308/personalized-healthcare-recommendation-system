# evaluation.py

"""
Evaluation Module for Personalized Healthcare Recommendation System

This module contains functions for evaluating the performance of predictive models and
the recommendation engine using appropriate metrics.

Techniques Used:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        self.models = {}
        self.data = None

    def load_data(self, filepath):
        """
        Load test data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        self.data = pd.read_csv(filepath)
        return self.data

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a file.
        
        :param model_name: str, name to assign to the loaded model
        :param filepath: str, path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)

    def evaluate_model(self, model_name, target_column):
        """
        Evaluate the specified model using various metrics.
        
        :param model_name: str, name of the model to evaluate
        :param target_column: str, name of the target column in the test data
        :return: dict, evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded.")
        
        model = self.models[model_name]
        X_test = self.data.drop(columns=[target_column])
        y_test = self.data[target_column]
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        return metrics

if __name__ == "__main__":
    test_data_filepath = 'data/processed/preprocessed_patient_data_test.csv'
    model_filepath = 'models/random_forest_model.pkl'
    target_column = 'target'

    evaluator = ModelEvaluation()
    evaluator.load_data(test_data_filepath)
    evaluator.load_model('random_forest', model_filepath)
    
    metrics = evaluator.evaluate_model('random_forest', target_column)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
