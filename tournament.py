import numpy as np
import pandas as pd
from fairlearn.metrics import selection_rate, MetricFrame
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def csv_to_sklearn_dataset():
    """
    load dataset
    :return:
    """
    # Load CSV file as pandas DataFrame
    df = pd.read_csv('dataset.csv')

    # Extract features and target variable
    X = df[['age', 'gender', 'race', 'blood_pressure', 'bmi']]
    y = df['readmission']

    # Create sklearn dataset object
    dataset = load_iris(as_frame=True)
    dataset.data = X
    dataset.target = y

    return dataset


param_dict = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'saga']
}


def create_individual():
    """
    create an individual
    :return:
    """
    return [np.random.choice(param_dict[key]) for key in param_dict.keys()]


def fitness(individual, x, y_true, sensitive_features):
    """
    get the fitness of an individual
    the fitness is the accuracy - selection rate difference
    selection rate difference is the difference of selection rate between different groups
    if the selection rate difference is large, then the fairness is low
    we consider the fairness as the negative of selection rate difference, sensitive features are age, race and gender
    :param individual:
    :param x:
    :param y_true:
    :param sensitive_features:
    :return:
    """
    clf = LogisticRegression(
        penalty=individual[0],
        C=individual[1],
        solver=individual[2],
        max_iter=3000
    )
    metrics = {
        "accuracy": accuracy_score,
        "selection rate": selection_rate,
    }
    clf.fit(x, y_true)
    y_pred = clf.predict(x)

    acc = 0
    sr_diff_all = 0
    # sensitive_features has gender,age,race
    for sensitive_feature in sensitive_features:
        gm = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feature)
        acc = gm.overall["accuracy"]
        # get the difference of selection rate between  groups
        sr_diff = gm.difference(method="between_groups")["selection rate"]
        sr_diff_all += sr_diff

    fitness = (acc - sr_diff_all * 10)  # maximize accuracy, minimize selection rate difference
    return fitness,


def tournament_selection(population, scores, k=6):
    """
    select k individuals from the population based on their scores
    select the individual with the largest score
    :param population:
    :param scores:
    :param k:
    :return:
    """
    selected = []
    visited_indices = set()

    for _ in range(k):
        max_score = None
        max_index = None

        # find the largest unvisited score
        for i in range(len(scores)):
            if i not in visited_indices and (max_score is None or scores[i] > max_score):
                max_score = scores[i]
                max_index = i

        # add the selected individual to the list
        selected.append(population[max_index])
        visited_indices.add(max_index)

    return selected


def uniform_crossover(parent1, parent2):
    """
    Uniform crossover
    if the random number is less than 0.5, then take the gene from parent1
    else take the gene from parent2
    :param parent1:
    :param parent2:
    :return:
    """
    offspring = []
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            offspring.append(parent1[i])
        else:
            offspring.append(parent2[i])
    return offspring


def mutation(individual):
    """
    Mutate a single individual
    if the random number is less than the mutation rate, then mutate
    :param individual:
    :return:
    """
    mutated_individual = individual.copy()
    param_idx = np.random.randint(len(param_dict.keys()))
    key = list(param_dict.keys())[param_idx]
    mutated_individual[param_idx] = np.random.choice(param_dict[key])
    return mutated_individual


POPULATION_SIZE = 50
MAX_GENERATIONS = 10
MUTATION_RATE = 0.2
gender = None
age = None
race = None

data = csv_to_sklearn_dataset()

# hot encode
x = pd.get_dummies(data.data)

gender = data.data['gender']
age = data.data['age']
race = data.data['race']

y_true = data.target

population = [create_individual() for _ in range(POPULATION_SIZE)]
scores = [fitness(individual, x, y_true, [gender, age, race]) for individual in population]
print(population)
print(scores)
# population maximum value
max_score = max(scores)
'''
plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True


list_scores = [element for tupl in scores for element in tupl]

length = len(scores)
x = [a for a in range(0,length)]
y = list_scores
print(list_scores)

plt.plot(x, y)
plt.axis([0, 51, -10, 20])


plt.show()
'''
for i in range(MAX_GENERATIONS):
    print(f"Generation {i + 1}")
    selected = tournament_selection(population, scores)

    offspring = []
    while len(offspring) < POPULATION_SIZE:
        p1 = np.random.choice(len(selected))
        p2 = np.random.choice(len(selected))
        parent1 = selected[p1]
        parent2 = selected[p2]

        if parent1 != parent2:
            child = uniform_crossover(parent1, parent2)
            if np.random.random() < MUTATION_RATE:
                child = mutation(child)
            offspring.append(child)
    population = offspring
    scores = [fitness(individual, x, y_true, [gender, age, race]) for individual in population]
    print(scores)

best_individual_idx = np.argmax(scores)
best_individual = population[best_individual_idx]
print(f"Best individual: {best_individual}")
print(f"Score: {scores[best_individual_idx][0]}")

# before optimization
print(f"before optimization: {max_score}")