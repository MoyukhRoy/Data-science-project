import pandas as pd
from pulp import *

data = pd.read_csv('DIY Dataset/diet.csv')

foods = list(data['Foods'])

price = dict(zip(foods, data['Price/Serving ($)']))
calories = dict(zip(foods, data['Calories']))
cholesterol = dict(zip(foods, data['Cholesterol (mg)']))
total_fat = dict(zip(foods, data['Total_Fat (g)']))
sodium = dict(zip(foods, data['Sodium (mg)']))
carbohydrates = dict(zip(foods, data['Carbohydrates (g)']))
fiber = dict(zip(foods, data['Dietary_Fiber (g)']))
protein = dict(zip(foods, data['Protein (g)']))
vit_a = dict(zip(foods, data['Vit_A (IU)']))
vit_c = dict(zip(foods, data['Vit_C (IU)']))
calcium = dict(zip(foods, data['Calcium (mg)']))
iron = dict(zip(foods, data['Iron (mg)']))

food_vars = LpVariable.dicts("Food", foods, lowBound=0, cat='Continuous')

prob = LpProblem("OptimalDiet", LpMinimize)
prob += lpSum([price[i] * food_vars[i] for i in foods])

prob += lpSum([calories[i] * food_vars[i] for i in foods]) >= 1500.0, "Calories_min"
prob += lpSum([calories[i] * food_vars[i] for i in foods]) <= 2500.0, "Calories_max"

prob.solve()

print("Status:", LpStatus[prob.status])

for food in food_vars:
    if food_vars[food].value() > 0:
        print(food, "=", food_vars[food].value())
