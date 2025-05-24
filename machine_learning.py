# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('data_with_all_scores_scaled.csv')

# Show first few rows to understand the data
print("First 5 rows of the data:")
print(data.head())

# Define target and features
target = 'SustainabilityScore'

# Include missing_market_value as a feature (convert boolean to int)
data['missing_market_value'] = data['missing_market_value'].astype(int)

# Features for the model
features = ['Age', 'Transfer History Count', 'MarketValue', 'missing_market_value', 'Age_norm',
            'MarketValue_norm', 'Transfers_norm', 'Nationality_norm', 'ClubTier', 'ClubTier_norm',
            'InputScore', 'ASR']

# Check for missing values in these columns and drop rows with missing data
print("\nMissing values per feature before dropping rows:")
print(data[features + [target]].isnull().sum())

data = data.dropna(subset=features + [target])

# Separate features and target
X = data[features]
y = data[target]

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor and train it
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score (R2): {r2:.4f}")

# Plot actual vs predicted Sustainability Score to visualize performance
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Sustainability Score")
plt.ylabel("Predicted Sustainability Score")
plt.title("Actual vs Predicted Sustainability Score")
plt.grid(True)
plt.show()

# Plot feature importances with explanations
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10,6))
colors = sns.color_palette('viridis', len(features))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Predicting Sustainability Score')
plt.grid(axis='x')
plt.show()

print("\nFeature importance shows which features the model relied on most to make predictions. "
      "Higher values mean the feature contributes more to the prediction.")

# Visualize correlation heatmap between features and target
plt.figure(figsize=(12,10))
corr_matrix = data[features + [target]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix Between Features and Sustainability Score")
plt.show()

print("\nCorrelation matrix helps identify linear relationships between variables. "
      "Strong positive or negative correlations might help model prediction.")

# --- Normalization info ---

# Let's assume you have these min and max values from your dataset or you can calculate them:
age_min = data['Age'].min()
age_max = data['Age'].max()

marketvalue_min = data['MarketValue'].min()
marketvalue_max = data['MarketValue'].max()

transfers_min = data['Transfer History Count'].min()
transfers_max = data['Transfer History Count'].max()

# Define a function to normalize any value:
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# New player raw data (example):
new_player_raw = {
    'Age': 24,
    'Transfer History Count': 3,
    'MarketValue': 15000000,
    'missing_market_value': 0,  # Known market value
    'Nationality_norm': 1.0,  # Assuming categorical encoded normalized value given directly
    'ClubTier': 1,
    'ClubTier_norm': 0.4,
    'InputScore': 1.4,
    'ASR': 7.0
}

# Calculate normalized values:
new_player_raw['Age_norm'] = normalize(new_player_raw['Age'], age_min, age_max)
new_player_raw['MarketValue_norm'] = normalize(new_player_raw['MarketValue'], marketvalue_min, marketvalue_max)
new_player_raw['Transfers_norm'] = normalize(new_player_raw['Transfer History Count'], transfers_min, transfers_max)

# Create DataFrame for model input
model_features = ['Age', 'Transfer History Count', 'MarketValue', 'missing_market_value', 'Age_norm',
                  'MarketValue_norm', 'Transfers_norm', 'Nationality_norm', 'ClubTier', 'ClubTier_norm',
                  'InputScore', 'ASR']

new_player_df = pd.DataFrame([{feat: new_player_raw[feat] for feat in model_features}])

# Predict Sustainability Score with the trained model
predicted_sustainability = rf_model.predict(new_player_df)[0]

# Print the results with raw and normalized values
print("New player data:")
print(f"Age (raw): {new_player_raw['Age']}, Age (normalized): {new_player_raw['Age_norm']:.3f}")
print(f"MarketValue (raw): {new_player_raw['MarketValue']}, MarketValue (normalized): {new_player_raw['MarketValue_norm']:.3f}")
print(f"Transfer History Count (raw): {new_player_raw['Transfer History Count']}, Transfers_norm: {new_player_raw['Transfers_norm']:.3f}")

print(f"\nPredicted Sustainability Score: {predicted_sustainability:.3f}")
