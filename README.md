# Bio_mass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from deap import base, creator, tools, algorithms
import random

# data frame and training
data = pd.DataFrame({
    'Particle Size (mm)': [0.2775, 0.75, 4.8],
    'Biomass %': [70, 80, 70],
    'Calorific Value (kcal/kg)': [4122.16, 4122.6, 4150.0],  # Example calorific values
    'Ash Content (%)': [13.6, 13.6, 14.0]  # Example ash content values
})


X = data[['Particle Size (mm)', 'Biomass %']]
y = data['Calorific Value (kcal/kg)']


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


model = LinearRegression()
model.fit(X_poly, y)


input_sample = np.array([[0.2775, 60]])
input_poly = poly.transform(input_sample)
predicted_calorific_value = model.predict(input_poly)[0]


print(f"Predicted Calorific Value at Particle Size 0.2775 and Biomass 60%: {predicted_calorific_value}")



creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximize calorific value, minimize ash content
creator.create("Individual", list, fitness=creator.FitnessMulti)



def evaluate(individual):
    particle_size, biomass_percentage = individual

    input_sample = np.array([[particle_size, biomass_percentage]])
    input_poly = poly.transform(input_sample)
    calorific_value_pred = model.predict(input_poly)[0]


    ash_content_pred = 13.6

    return calorific_value_pred, ash_content_pred

# calling tool box
toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)


toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.2775, 60], up=[4.8, 80], eta=1.0, indpb=0.2)

toolbox.register("select", tools.selNSGA2)
toolbox.register("attribute", random.uniform, 0.2775, 4.8)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attribute, toolbox.attribute), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# total population
population = toolbox.population(n=100)
algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=40, verbose=True)

# population
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
pareto_calorific = [evaluate(ind)[0] for ind in pareto_front]
pareto_ash = [evaluate(ind)[1] for ind in pareto_front]

# graph
plt.figure(figsize=(10, 6))
plt.scatter(pareto_ash, pareto_calorific, c='blue', label='Pareto Front')
plt.scatter(13.6, predicted_calorific_value, c='red', label='Predicted Value at Biomass 60%')
plt.xlabel('Ash Content (%)')
plt.ylabel('Calorific Value (kcal/kg)')
plt.title('Pareto Front of Multi-Objective Optimization (NSGA-II)')
plt.legend()
plt.show()

# So basically this is project of machine learning develop in JC. bose college and with the help of faculty
