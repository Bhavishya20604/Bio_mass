import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Step 1: Load and prepare the dataset
data = pd.read_csv(r"C:\Users\HP\Downloads\Khushi_data.csv")  # Update with actual file path

# Calculate mean calorific value and ash content
data['Calorific_Value_Mean'] = data[['Calorific Value 1 (kcal/kg)', 'Calorific Value 2 (kcal/kg)', 'Calorific Value 3 (kcal/kg)']].mean(axis=1)
data['Ash_Content_Mean'] = data[['Ash Content 1 (%)', 'Ash Content 2 (%)', 'Ash Content 3 (%)']].mean(axis=1)

# Drop original columns to use only the mean values
data.drop(['Calorific Value 1 (kcal/kg)', 'Calorific Value 2 (kcal/kg)', 'Calorific Value 3 (kcal/kg)',
           'Ash Content 1 (%)', 'Ash Content 2 (%)', 'Ash Content 3 (%)'], axis=1, inplace=True)

# Apply one-hot encoding to the 'Biomass_Type' column
data = pd.get_dummies(data, columns=['Biomass_Type'], drop_first=True)

# Prepare features and target variables
X = data.drop(['Calorific_Value_Mean', 'Ash_Content_Mean'], axis=1)
y_calorific = data['Calorific_Value_Mean']
y_ash = data['Ash_Content_Mean']

# Train-test split
X_train, X_test, y_train_calorific, y_test_calorific, y_train_ash, y_test_ash = train_test_split(
    X, y_calorific, y_ash, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define the objective function for Differential Evolution optimization (Random Forest)
def objective_rf(params):
    n_estimators, max_depth, min_samples_split = int(params[0]), int(params[1]), int(params[2])
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    rf.fit(X_train_scaled, y_train_calorific)
    y_pred = rf.predict(X_test_scaled)
    mse = mean_squared_error(y_test_calorific, y_pred)
    return mse

# Step 4: Differential Evolution Optimization
bounds = [(50, 500), (5, 30), (2, 20)]  # n_estimators, max_depth, min_samples_split
result = differential_evolution(objective_rf, bounds, maxiter=10, popsize=10, mutation=(0.5, 1), recombination=0.7)
best_rf_params = result.x
print(f"Best Random Forest hyperparameters: {best_rf_params}")

# Step 5: Train Random Forest with optimized hyperparameters
rf_optimized = RandomForestRegressor(n_estimators=int(best_rf_params[0]),
                                     max_depth=int(best_rf_params[1]),
                                     min_samples_split=int(best_rf_params[2]),
                                     random_state=42)
rf_optimized.fit(X_train_scaled, y_train_calorific)

# Save the trained Random Forest model
joblib.dump(rf_optimized, 'random_forest_model.pkl')

# Step 6: Model Evaluation - Evaluate Random Forest Model Performance
y_pred_rf = rf_optimized.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test_calorific, y_pred_rf)
rf_mae = mean_absolute_error(y_test_calorific, y_pred_rf)
rf_r2 = r2_score(y_test_calorific, y_pred_rf)

print(f"Random Forest Model Performance:\nMSE: {rf_mse}, MAE: {rf_mae}, R²: {rf_r2}")

# Step 7: Train Gradient Boosting Model
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train_calorific)

# Step 8: Train XGBoost Model
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
xgb.fit(X_train_scaled, y_train_calorific)

# Step 9: Model Evaluation using Cross-validation
models = [rf_optimized, gb, xgb]
model_names = ['Random Forest', 'Gradient Boosting', 'XGBoost']
cv_scores = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for model in models:
    mse = -cross_val_score(model, X, y_calorific, cv=kf, scoring='neg_mean_squared_error').mean()
    cv_scores.append(mse)

# Step 10: Plot Model Comparison (Cross-Validation MSE Scores)
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=cv_scores, palette="viridis", hue=model_names)
plt.title("Model Comparison - Cross-Validation MSE Scores")
plt.xlabel("Model")
plt.ylabel("Mean Squared Error (MSE)")
plt.tight_layout()
plt.savefig("model_comparison_mse.png", dpi=300)  # Save with high quality (300 DPI)
plt.show()

# Step 11: Plot Feature Importance (for Random Forest)
explainer = shap.TreeExplainer(rf_optimized)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300)
plt.show()

# Step 12: Hyperparameter Tuning for Gradient Boosting using GridSearchCV
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5)
grid_search_gb.fit(X_train_scaled, y_train_calorific)

print(f"Best Gradient Boosting hyperparameters: {grid_search_gb.best_params_}")

# Step 13: Train Gradient Boosting with optimal parameters from GridSearchCV
gb_optimal = grid_search_gb.best_estimator_

# Save Gradient Boosting model
joblib.dump(gb_optimal, 'gradient_boosting_model.pkl')

# Step 14: Evaluate Gradient Boosting Model Performance
y_pred_gb = gb_optimal.predict(X_test_scaled)
gb_mse = mean_squared_error(y_test_calorific, y_pred_gb)
gb_mae = mean_absolute_error(y_test_calorific, y_pred_gb)
gb_r2 = r2_score(y_test_calorific, y_pred_gb)

print(f"Gradient Boosting Model Performance:\nMSE: {gb_mse}, MAE: {gb_mae}, R²: {gb_r2}")

# Step 15: Hyperparameter Tuning for XGBoost using RandomizedSearchCV
param_dist = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
random_search_xgb = RandomizedSearchCV(XGBRegressor(random_state=42), param_dist, cv=5, n_iter=5, random_state=42)
random_search_xgb.fit(X_train_scaled, y_train_calorific)

print(f"Best XGBoost hyperparameters: {random_search_xgb.best_params_}")

# Step 16: Train XGBoost with optimal parameters from RandomizedSearchCV
xgb_optimal = random_search_xgb.best_estimator_

# Save XGBoost model
joblib.dump(xgb_optimal, 'xgboost_model.pkl')

# Step 17: Evaluate XGBoost Model Performance
y_pred_xgb = xgb_optimal.predict(X_test_scaled)
xgb_mse = mean_squared_error(y_test_calorific, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test_calorific, y_pred_xgb)
xgb_r2 = r2_score(y_test_calorific, y_pred_xgb)

print(f"XGBoost Model Performance:\nMSE: {xgb_mse}, MAE: {xgb_mae}, R²: {xgb_r2}")
