Weather Prediction Analysis and Modeling

This project demonstrates an end-to-end pipeline for weather data preprocessing, feature engineering, and prediction modeling using Python and machine learning libraries. The model predicts the probability of sunny weather based on historical weather data.

Table of Contents

Overview
Requirements
Dataset
Pipeline Description
Modeling and Evaluation
How to Run
Results
License
Overview

The script performs the following tasks:

Cleans and preprocesses weather data.
Handles missing values and outliers.
Creates new features for better prediction accuracy.
Trains a Random Forest Classifier to predict sunny weather.
Evaluates and visualizes the model's performance using accuracy and feature importance.
Dataset

The script uses a CSV file (data.csv) containing weather data with the following columns:

precipitation: Precipitation levels.
temp_max: Maximum temperature.
temp_min: Minimum temperature.
wind: Wind speed.
weather: Weather description (e.g., sunny, cloudy, etc.).
Ensure the dataset is placed in the same directory as the script.

Pipeline Description

Data Cleaning:
Handles missing values in numerical columns using a rolling median and column median as fallback.
Removes duplicates and fixes invalid temperature values where temp_max < temp_min.
Feature Engineering:
Creates a new feature: temp_range (difference between temp_max and temp_min).
Converts the weather column to a binary target variable is_sunny.
Outlier Handling:
Uses the Interquartile Range (IQR) method to cap values in numerical columns.
Model Training:
Splits data into training and testing sets.
Scales features using StandardScaler.
Trains a Random Forest Classifier on the scaled features.
Prediction and Evaluation:
Predicts sunny weather probabilities for the entire dataset.
Visualizes results using rolling accuracy plots and prediction error distribution.
Modeling and Evaluation

Feature Correlation: Displays correlation between input features and the target variable.
Feature Importance: Lists the importance of each feature in predicting sunny weather.
Performance Metrics:
Accuracy
Precision
Recall
Visualization:
Rolling accuracy (e.g., 30-day window).
Prediction error distribution.
Results

The script provides:

Performance Metrics:
Accuracy, Precision, and Recall values for the model.
Visualizations:
Rolling accuracy plot to monitor performance over time.
Histogram of prediction errors.
Feature Insights:
Feature correlations and importance rankings.
Prediction Summary:
Breakdown of correct and incorrect predictions.
