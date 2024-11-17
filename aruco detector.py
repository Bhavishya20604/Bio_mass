from scipy.stats import qmc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import random



# Step 1: Generate Data using Latin Hypercube Sampling
np.random.seed(42)

# Define input levels for LHS
particle_sizes = [4.8, 2.775, 0.75]
biomass_types = ["Henna", "Shrubs", "Berries", "Henna+Shrubs", "Henna+Berries", "Shrubs+Berries"]
biomass_percentages = [60, 70, 80]
num_experiments = 24

# Generate LHS samples
particle_size_samples = np.random.choice(particle_sizes, num_experiments)
biomass_type_samples = np.random.choice(biomass_types, num_experiments)
biomass_percent_samples = np.random.choice(biomass_percentages, num_experiments)

# Create DataFrame
data = pd.DataFrame({
    'Particle Size (mm)': particle_size_samples,
    'Biomass_Type': biomass_type_samples,
    'Biomass %': biomass_percent_samples
})

# Step 2: Map Combination Percentages
def calculate_combination_percentage(row):
    biomass_pct = row['Biomass %']
    return biomass_pct / 2 if "+" in row['Biomass_Type'] else biomass_pct

data['Component %'] = data.apply(calculate_combination_percentage, axis=1)

# Step 3: Simulate Calorific Value and Ash Content
np.random.seed(0)
for col in ['Calorific Value 1 (kcal/kg)', 'Calorific Value 2 (kcal/kg)', 'Calorific Value 3 (kcal/kg)']:
    data[col] = np.random.uniform(3000, 4800, size=len(data))
for col in ['Ash Content 1 (%)', 'Ash Content 2 (%)', 'Ash Content 3 (%)']:
    data[col] = np.random.uniform(5, 15, size=len(data))

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(data['Particle Size (mm)'], kde=True, bins=10)
plt.title('Distribution of Particle Size')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Biomass_Type', y='Biomass %', data=data)
plt.title('Biomass % by Biomass Type')
plt.show()

# Step 5: Preprocessing
label_encoder = LabelEncoder()
data['Biomass_Type_encoded'] = label_encoder.fit_transform(data['Biomass_Type'])
scaler = StandardScaler()
data[['Particle Size (mm)', 'Biomass %']] = scaler.fit_transform(data[['Particle Size (mm)', 'Biomass %']])

# Step 6: Define Feature and Target Variables
X = data[['Particle Size (mm)', 'Biomass_Type_encoded', 'Biomass %']]
biomass_type_value = individual_df['Biomass_Type_encoded'].iloc[0]

y_calorific = data[['Calorific Value 1 (kcal/kg)', 'Calorific Value 2 (kcal/kg)', 'Calorific Value 3 (kcal/kg)']].mean(axis=1)
y_ash = data[['Ash Content 1 (%)', 'Ash Content 2 (%)', 'Ash Content 3 (%)']].mean(axis=1)

# Step 7: Train Random Forest Model for Calorific Value and Ash Content
rf_calorific = RandomForestRegressor(random_state=42)
rf_ash = RandomForestRegressor(random_state=42)
param_dist = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
rf_calorific_best = RandomizedSearchCV(rf_calorific, param_dist, n_iter=10, cv=3, random_state=42).fit(X, y_calorific)
rf_ash_best = RandomizedSearchCV(rf_ash, param_dist, n_iter=10, cv=3, random_state=42).fit(X, y_ash)

# Step 8: Multi-Objective Optimization using NSGA-II
# Check if creator classes already exist
if 'FitnessMulti' not in dir(creator):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
if 'Individual' not in dir(creator):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def initIndividual():
    return [random.uniform(0, 1) for _ in range(len(X.columns))]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, initIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    # Confirm that 'Biomass_Type' is in the columns of X
    if 'Biomass_Type' not in X.columns:
        raise KeyError("'Biomass_Type' not found in feature set columns.")

    # Convert the individual to a DataFrame with correct column names
    individual_df = pd.DataFrame([individual], columns=X.columns)

    # Map the continuous value of 'Biomass_Type' to the nearest category
    biomass_type_value = individual_df['Biomass_Type'].iloc[0]
    categories = label_encoder.classes_
    index = int(np.floor(biomass_type_value * len(categories)))
    index = min(max(index, 0), len(categories) - 1)
    individual_df['Biomass_Type'] = categories[index]

    # Ensure individual values are real numbers and preprocess with scaler
    individual_df = individual_df.apply(np.real)

    # Scale individual and match column names and order to X
    scaled_individual_df = pd.DataFrame(scaler.transform(individual_df), columns=X.columns)

    # Reorder columns to match the order used in model training
    scaled_individual_df = scaled_individual_df[X.columns]

    # Make predictions using the trained models
    calorific_pred = rf_calorific_best.predict(scaled_individual_df)[0]
    ash_pred = rf_ash_best.predict(scaled_individual_df)[0]

    return calorific_pred, ash_pred


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Initializing the population and running the genetic algorithm
population = toolbox.population(n=100)
algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=40, verbose=True)

# Step 9: Extract Pareto Front and Plot
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
pareto_calorific = [evaluate(ind)[0] for ind in pareto_front]
pareto_ash = [evaluate(ind)[1] for ind in pareto_front]

plt.figure(figsize=(10, 6))
plt.scatter(pareto_ash, pareto_calorific, c='blue', label='Pareto Front')
plt.xlabel('Ash Content (%)')
plt.ylabel('Calorific Value (kcal/kg)')
plt.title('Pareto Front of Multi-Objective Optimization')
plt.legend()
plt.show()

