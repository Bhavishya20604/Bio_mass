import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


data = {
    'Sample No.': range(1, 14),
    'Cow Dung %': [40, 30, 20, 40, 30, 20, 40, 30, 20, 0, 0, 0, 100],
    'Biomass Type': ['Henna Berries', 'Henna Berries', 'Henna Berries',
                     'Henna Leaves', 'Henna Leaves', 'Henna Leaves',
                     'Henna Shrubs', 'Henna Shrubs', 'Henna Shrubs',
                     'Henna Berries', 'Henna Shrubs', 'Henna Leaves', 'Cow Dung'],
    'Avg. Calorific Value (kcal/kg)': [3863.88, 4089.10, 4562.57, 3758.27,
                                       4148.25, 3754.57, 3366.85, 3552.72,
                                       3513.26, 4802.30, 4305.18, 4502.29, 3282.25],
    'Avg. Ash Content %': [15.56, 11.28, 5.62, 18.20, 14.91, 12.91,
                           5.14, 4.06, 3.09, 6.56, 3.32, 7.88, 32.06]
}


df = pd.DataFrame(data)


anova = ols('Q("Avg. Calorific Value (kcal/kg)") ~ Q("Cow Dung %")', data=df).fit()
anova_table = sm.stats.anova_lm(anova, typ=2)
print("ANOVA Table for Calorific Value:")
print(anova_table)


X = df['Cow Dung %']
Y = df['Avg. Calorific Value (kcal/kg)']

X = sm.add_constant(X)  
model = sm.OLS(Y, X).fit()

print("\nRegression Summary:")
print(model.summary())



plt.figure(figsize=(8, 6))
sns.lineplot(x='Cow Dung %', y='Avg. Calorific Value (kcal/kg)', hue='Biomass Type', data=df, marker="o")
plt.title('Calorific Value vs Cow Dung Percentage')
plt.ylabel('Avg. Calorific Value (kcal/kg)')
plt.xlabel('Cow Dung Percentage')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x='Cow Dung %', y='Avg. Ash Content %', hue='Biomass Type', data=df, marker="o")
plt.title('Ash Content vs Cow Dung Percentage')
plt.ylabel('Avg. Ash Content (%)')
plt.xlabel('Cow Dung Percentage')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df['Cow Dung %'], df['Avg. Calorific Value (kcal/kg)'], color='blue', label='Data Points')
plt.plot(df['Cow Dung %'], model.predict(X), color='red', label='Regression Line')
plt.title('Regression of Calorific Value on Cow Dung Percentage')
plt.ylabel('Avg. Calorific Value (kcal/kg)')
plt.xlabel('Cow Dung Percentage')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='Biomass Type', y='Avg. Calorific Value (kcal/kg)', data=df)
plt.title('Boxplot of Calorific Value by Biomass Type')
plt.ylabel('Avg. Calorific Value (kcal/kg)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='Biomass Type', y='Avg. Ash Content %', data=df)
plt.title('Boxplot of Ash Content by Biomass Type')
plt.ylabel('Avg. Ash Content (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
