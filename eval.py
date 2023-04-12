import clf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import permutations
import random
# Load dataset
dataset = pd.read_csv("dataset.csv")

# Preprocessing
le = LabelEncoder()
dataset["gender"] = le.fit_transform(dataset["gender"])
dataset["race"] = le.fit_transform(dataset["race"])
dataset["blood_pressure"] = le.fit_transform(dataset["blood_pressure"])
X = dataset.drop("readmission", axis=1)
y = dataset["readmission"]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Define fitness function
def fitness(chromosome):
    # Decode chromosome
    features = np.array(chromosome[:4])
    selected_features = [features[i] for i in range(len(features)) if chromosome[i] == 1]
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
    group_0 = [i for i in range(len(y_test)) if y_test[i] == 0]
    group_1 = [i for i in range(len(y_test)) if y_test[i] == 1]
    group_0_present = all(i in group_0 for i in group_0)
    group_1_present = all(i in group_1 for i in group_1)
    if group_0_present and group_1_present:
        group_0_acc = accuracy_score(y_test[group_0], y_pred[group_0])
        group_1_acc = accuracy_score(y_test[group_1], y_pred[group_1])
        return (group_0_acc + group_1_acc) / 2
    else:
        return 0



# Define chromosome representation

# Genetic algorithm


#set pop_size = 20, chrom_length = 4
def launch_chro(pop_size, chrom_length):
    
    range_list = list(range(1,chrom_length+1))
    population = list(permutations(range_list))
    population = random.sample(population, pop_size)

    
    
    return population     

def permutation_swap (individual):
    """Mutate a permutation"""

    mutant = individual.copy()
    chrom_len = range(len(mutant))
    x, y = random.sample(chrom_len, 2)
    mutant[x], mutant[y] = mutant[y], mutant[x]
    
    return mutant

print(launch_chro(20,4))

def permutation_cut_and_crossfill (parent1, parent2):
    """cut-and-crossfill crossover for permutation representations"""

    offspring1 = []
    offspring2 = []

    
    # student code begin
    
    chrom_length = len(parent1)
    
    cross_point = np.random.randint(1,chrom_length-1)
    crossed1 = []
    crossed2 = []
    for i in parent2[cross_point:]:
        if i not in parent1[:cross_point]:
            crossed1.append(i)
        else:
            continue
    for a in parent2[:cross_point]:
        if a not in parent1[:cross_point]:
            crossed1.append(a)
        else:
            continue
    
    offspring1 = parent1[:cross_point] + crossed1
    
    for j in parent1[cross_point:]:
        if j not in parent2[:cross_point]:
            crossed2.append(j)
        else:
            continue
    for b in parent1[:cross_point]:
        if b not in parent2[:cross_point]:
            crossed2.append(b)
        else:
            continue
    offspring2 = parent2[:cross_point] + crossed2
                
    # student code end
    
    return offspring1, offspring2

def MPS(fitness, mating_pool_size):
    """Multi-pointer selection (MPS)"""

    selected_to_mate = []

    # student code starts


    # Create a list of cumulative sum of the fitness scores
    cumulative_sum = [sum(fitness[:i + 1]) for i in range(len(fitness))]
    # Generate mating pool
    pointer = random.uniform(0, 1/mating_pool_size)   
    i = 0
    while cumulative_sum[i] <= mating_pool_size:
        while pointer > cumulative_sum[i]:
            if pointer < cumulative_sum[i+1]:
                selected_to_mate.append(cumulative_sum[i+1])
            i += 1
            pointer += 1/mating_pool_size


    # student code ends
    
    return selected_to_mate                