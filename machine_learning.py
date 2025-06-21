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


