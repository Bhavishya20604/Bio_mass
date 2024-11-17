import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deap import base, creator, tools, algorithms
import random


# Step 1: Latin Hypercube Sampling (LHS) Setup
def generate_lhs_samples(num_samples, particle_sizes, biomass_types, biomass_percentages):
    """ Generate LHS samples for particle sizes, biomass types, and biomass percentages. """
    particle_size_samples = np.random.choice(particle_sizes, num_samples)
    biomass_type_samples = np.random.choice(biomass_types, num_samples)
    biomass_percent_samples = np.random.choice(biomass_percentages, num_samples)

    return pd.DataFrame({
        'Particle Size (mm)': particle_size_samples,
        'Biomass_Type': biomass_type_samples,
        'Biomass %': biomass_percent_samples
    })


# Step 2: Mapping Combination Percentages
def calculate_combination_percentage(row):
    """ Calculate combination percentage based on biomass type. """
    biomass_pct = row['Biomass %']
    return biomass_pct / 2 if "+" in row['Biomass_Type'] else biomass_pct


# Step 3: Simulating Calorific Value and Ash Content
def simulate_calorific_and_ash_content(data):
    """ Simulate calorific value and ash content for the dataset. """
    np.random.seed(0)
    for col in ['Calorific Value 1 (kcal/kg)', 'Calorific Value 2 (kcal/kg)', 'Calorific Value 3 (kcal/kg)']:
        data[col] = np.random.uniform(3000, 4800, size=len(data))
    for col in ['Ash Content 1 (%)', 'Ash Content 2 (%)', 'Ash Content 3 (%)']:
        data[col] = np.random.uniform(5, 15, size=len(data))
    return data


# Step 4: Exploratory Data Analysis (EDA)
def plot_eda(data):
    """ Plot the distribution of particle size and biomass percentage by type. """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Particle Size (mm)'], kde=True, bins=10)
    plt.title('Distribution of Particle Size')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Biomass_Type', y='Biomass %', data=data)
    plt.title('Biomass % by Biomass Type')
    plt.show()


# Step 5: Preprocessing
def preprocess_data(data):
    """ Preprocess data by encoding categorical variables and scaling. """
    label_encoder = LabelEncoder()
    data['Biomass_Type_encoded'] = label_encoder.fit_transform(data['Biomass_Type'])

    scaler = StandardScaler()
    columns_to_scale = ['Particle Size (mm)', 'Biomass %']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data, label_encoder, scaler


# Step 6: Train Random Forest Models
def train_rf_models(X, y_calorific, y_ash):
    """ Train random forest models for calorific value and ash content prediction. """
    rf_calorific = RandomForestRegressor()
    rf_calorific.fit(X, y_calorific)

    rf_ash = RandomForestRegressor()
    rf_ash.fit(X, y_ash)

    return rf_calorific, rf_ash


# Step 7: Multi-objective Optimization using DEAP
def setup_deap_ga(evaluate_func):
    """ Set up DEAP genetic algorithm. """
    creator.create("FitnessMulti", base.Fitness,
                   weights=(1.0, -1.0))  # Multi-objective, maximize first and minimize second
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate_func)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=0.1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    return toolbox


# Step 8: Evaluation Function for Multi-Objective Optimization
def evaluate(individual, X, rf_calorific_best, rf_ash_best, label_encoder, scaler, columns_to_scale):
    """ Evaluate function for multi-objective optimization. """
    # Ensure individual is a list with correct shape
    individual_df = pd.DataFrame([individual], columns=X.columns)

    # Map continuous biomass type to nearest category
    biomass_type_value = individual_df['Biomass_Type_encoded'].iloc[0]
    categories = label_encoder.classes_
    index = int(np.floor(biomass_type_value * len(categories)))
    index = min(max(index, 0), len(categories) - 1)
    individual_df['Biomass_Type_encoded'] = categories[index]

    # Scale 'Particle Size' and 'Biomass %'
    scaled_values = scaler.transform(individual_df[columns_to_scale])
    scaled_individual_df = pd.DataFrame(scaled_values, columns=columns_to_scale)

    # Reconstruct the DataFrame to match training features
    scaled_individual_df['Biomass_Type_encoded'] = individual_df['Biomass_Type_encoded'].values
    scaled_individual_df = scaled_individual_df[X.columns]

    # Make predictions
    calorific_pred = rf_calorific_best.predict(scaled_individual_df)[0]
    ash_pred = rf_ash_best.predict(scaled_individual_df)[0]

    return calorific_pred, ash_pred


# Step 9: Main Execution Logic
def run_ga():
    # Generate LHS samples
    particle_sizes = [4.8, 2.775, 0.75]
    biomass_types = ["Henna", "Shrubs", "Berries", "Henna+Shrubs", "Henna+Berries", "Shrubs+Berries"]
    biomass_percentages = [60, 70, 80]
    num_experiments = 24

    data = generate_lhs_samples(num_experiments, particle_sizes, biomass_types, biomass_percentages)

    # Apply combination percentage and simulate calorific & ash content
    data['Component %'] = data.apply(calculate_combination_percentage, axis=1)
    data = simulate_calorific_and_ash_content(data)

    # EDA
    plot_eda(data)

    # Preprocess data
    X = data[['Particle Size (mm)', 'Biomass_Type_encoded', 'Biomass %']]
    y_calorific = data[
        ['Calorific Value 1 (kcal/kg)', 'Calorific Value 2 (kcal/kg)', 'Calorific Value 3 (kcal/kg)']].mean(axis=1)
    y_ash = data[['Ash Content 1 (%)', 'Ash Content 2 (%)', 'Ash Content 3 (%)']].mean(axis=1)

    data, label_encoder, scaler = preprocess_data(data)

    # Train Random Forest models
    rf_calorific, rf_ash = train_rf_models(X, y_calorific, y_ash)

    # Set up DEAP GA
    toolbox = setup_deap_ga(
        lambda ind: evaluate(ind, X, rf_calorific, rf_ash, label_encoder, scaler, ['Particle Size (mm)', 'Biomass %']))

    # Initialize population and run the GA
    population = tools.initRepeat(list, creator.Individual, n=100)
    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=40, verbose=True)

    # Extract Pareto Front and Plot
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    pareto_calorific = [
        evaluate(ind, X, rf_calorific, rf_ash, label_encoder, scaler, ['Particle Size (mm)', 'Biomass %'])[0] for ind in
        pareto_front]
    pareto_ash = [evaluate(ind, X, rf_calorific, rf_ash, label_encoder, scaler, ['Particle Size (mm)', 'Biomass %'])[1]
                  for ind in pareto_front]

    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_ash, pareto_calorific, c='blue', label='Pareto Front')
    plt.xlabel('Ash Content (%)')
    plt.ylabel('Calorific Value (kcal/kg)')
    plt.title('Pareto Front of Multi-Objective Optimization')
    plt.legend()
    plt.show()


# Run the GA process
run_ga()
