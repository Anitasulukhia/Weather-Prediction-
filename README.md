# Weather-Prediction

Weather Prediction Project

This project is a Python script that analyzes weather data and predicts sunny days using a machine learning model.

What This Script Does

Cleans and processes weather data.
Handles missing values and fixes data errors.
Creates new features to improve predictions.
Uses a Random Forest Classifier to predict sunny days.
Visualizes results and evaluates how well the model works.
Requirements

You need the following Python libraries installed:

pandas
numpy
scikit-learn
matplotlib
Install them by running:

pip install pandas numpy scikit-learn matplotlib
Input Data

The script uses a file called data.csv with the following columns:

precipitation: Amount of rain or snow.
temp_max: Maximum temperature.
temp_min: Minimum temperature.
wind: Wind speed.
weather: Weather description (e.g., sunny, cloudy).
Make sure data.csv is in the same folder as the script.

How It Works

Cleans the Data:
Fills in missing values.
Fixes any invalid temperature data.
Removes duplicates.
Creates New Features:
Adds temp_range (difference between max and min temperature).
Creates is_sunny, a column that marks if the weather is sunny.
Trains the Model:
Splits data into training and testing sets.
Scales the data for better model performance.
Trains a Random Forest Classifier to predict sunny weather.
Evaluates the Model:
Checks accuracy, precision, and recall.
Shows important features for predictions.
Visualizes results with charts.
