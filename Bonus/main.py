import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('data.csv')

# Data Cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Handle missing numerical values
numerical_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
for col in numerical_cols:
    # Use rolling median for temporal data
    df[col] = df[col].fillna(
        df[col].rolling(window=7, center=True, min_periods=1).median()
    )
    # If any remaining NaN, fill with column median
    df[col] = df[col].fillna(df[col].median())

# Handle outliers using IQR method
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Remove duplicates
df = df.drop_duplicates()


# Fix temp_max < temp_min cases
invalid_temps = df['temp_max'] < df['temp_min']
if invalid_temps.any():
    temp_max = df.loc[invalid_temps, 'temp_max'].copy()
    df.loc[invalid_temps, 'temp_max'] = df.loc[invalid_temps, 'temp_min']
    df.loc[invalid_temps, 'temp_min'] = temp_max

# Add temperature range feature
df['temp_range'] = df['temp_max'] - df['temp_min']

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Create target variable (is_sunny)
df['is_sunny'] = (df['weather'] == 'sun').astype(int)

# Select features for prediction
features = ['precipitation', 'temp_max', 'temp_min', 'wind', 'temp_range']
X = df[features]
y = df['is_sunny']

# Print feature correlations
print("\nFeature correlations with target:")
for feature in features:
    correlation = df[feature].corr(df['is_sunny'])
    print(f"{feature}: {correlation:.3f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Get predictions for all data
X_all_scaled = scaler.transform(X)
all_predictions = rf_model.predict_proba(X_all_scaled)[:, 1]  # Probability of sunny weather

# Add predictions to dataframe
df['predicted_sunny_prob'] = all_predictions

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Create confusion matrix visualization
def plot_prediction_analysis(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Moving average of accuracy
    window_size = 30  # 30-day window
    df['correct_prediction'] = (df['is_sunny'] == (df['predicted_sunny_prob'] > 0.5)).astype(int)
    df['rolling_accuracy'] = df['correct_prediction'].rolling(window=window_size).mean()

    ax1.plot(df.index, df['rolling_accuracy'], color='green')
    ax1.set_title(f'{window_size}-Day Rolling Accuracy')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)  # Reference line at 50%

    # Plot 2: Prediction Error Distribution
    prediction_error = df['predicted_sunny_prob'] - df['is_sunny']
    ax2.hist(prediction_error, bins=50, color='skyblue', edgecolor='black')
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    total_predictions = len(df)
    correct_predictions = df['correct_prediction'].sum()
    accuracy = correct_predictions / total_predictions

    print("\nPrediction Summary:")
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {accuracy:.2%}")

    # Analyze prediction patterns
    true_positives = ((df['is_sunny'] == 1) & (df['predicted_sunny_prob'] > 0.5)).sum()
    true_negatives = ((df['is_sunny'] == 0) & (df['predicted_sunny_prob'] <= 0.5)).sum()
    false_positives = ((df['is_sunny'] == 0) & (df['predicted_sunny_prob'] > 0.5)).sum()
    false_negatives = ((df['is_sunny'] == 1) & (df['predicted_sunny_prob'] <= 0.5)).sum()

    print("\nDetailed Breakdown:")
    print(f"Correctly predicted sunny days: {true_positives}")
    print(f"Correctly predicted non-sunny days: {true_negatives}")
    print(f"False sunny predictions: {false_positives}")
    print(f"Missed sunny days: {false_negatives}")

plot_prediction_analysis(df)

# Print metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = (df['predicted_sunny_prob'] > 0.5).astype(int)
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(df['is_sunny'], y_pred):.2f}")
print(f"Precision: {precision_score(df['is_sunny'], y_pred):.2f}")
print(f"Recall: {recall_score(df['is_sunny'], y_pred):.2f}")