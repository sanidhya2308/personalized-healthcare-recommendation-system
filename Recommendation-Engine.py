# recommendation_engine.py

"""
Recommendation Engine Module for Personalized Healthcare Recommendation System

This module contains functions for building and deploying a recommendation engine
that provides personalized healthcare interventions based on patient data.

Techniques Used:
- Collaborative filtering
- Content-based filtering
- Hybrid approaches
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import joblib

class RecommendationEngine:
    def __init__(self):
        """
        Initialize the RecommendationEngine class.
        """
        self.model = None
        self.patient_data = None

    def load_data(self, filepath):
        """
        Load patient data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        self.patient_data = pd.read_csv(filepath)
        return self.patient_data

    def train_collaborative_filtering(self, n_neighbors=5):
        """
        Train a collaborative filtering model.
        
        :param n_neighbors: int, number of neighbors for KNN
        """
        if self.patient_data is None:
            raise ValueError("Patient data not loaded.")
        
        # Assuming the patient_data contains patient IDs and their features/interactions
        features = self.patient_data.drop(columns=['patient_id'])
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.model.fit(features)
    
    def recommend(self, patient_id, n_recommendations=5):
        """
        Provide recommendations for a given patient.
        
        :param patient_id: int, ID of the patient
        :param n_recommendations: int, number of recommendations to return
        :return: list, recommended patient IDs
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        patient_idx = self.patient_data[self.patient_data['patient_id'] == patient_id].index[0]
        patient_features = self.patient_data.drop(columns=['patient_id']).iloc[patient_idx].values.reshape(1, -1)
        distances, indices = self.model.kneighbors(patient_features, n_neighbors=n_recommendations + 1)
        
        recommended_patient_ids = self.patient_data.iloc[indices.flatten()]['patient_id'].values
        return recommended_patient_ids[recommended_patient_ids != patient_id][:n_recommendations]

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        :param filepath: str, path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        :param filepath: str, path to the saved model
        """
        self.model = joblib.load(filepath)

if __name__ == "__main__":
    filepath = 'data/processed/preprocessed_patient_data.csv'
    
    recommendation_engine = RecommendationEngine()
    recommendation_engine.load_data(filepath)
    
    # Train the recommendation engine
    recommendation_engine.train_collaborative_filtering(n_neighbors=5)
    
    # Save the trained model
    model_filepath = 'models/recommendation_model.pkl'
    recommendation_engine.save_model(model_filepath)
    print(f"Trained model saved as {model_filepath}")
    
    # Load the trained model
    recommendation_engine.load_model(model_filepath)
    
    # Get recommendations for a specific patient
    patient_id = 1  # Replace with a valid patient ID from the dataset
    recommendations = recommendation_engine.recommend(patient_id, n_recommendations=5)
    print(f"Recommendations for patient {patient_id}: {recommendations}")
