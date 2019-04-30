# -------------------------------------------------------------------------- #

# Prática 2 - Implementação do Algoritmo Artificial Bee Colony(ABC)
# Aluno: Eduardo Luís Anselmo Batista
# Disciplina: Computação Natural
# Professor: Carmelo
# Engenharia da Computação - POLI-UPE - 2018.1

# -------------------------------------------------------------------------- #


import numpy as np
import random as rand
import matplotlib.pyplot as plt

from operator import attrgetter

#FOODSOURCE DENTRO DO ESPAÇO DE BUSCA
class FoodSource(object):
    trials = 0

    #INCIALIZADORES
    def __init__(self, initial_solution, initial_fitness):
        super(FoodSource, self).__init__()

        self.solution = initial_solution
        self.fitness = initial_fitness
        self.history_fitness = []

    #FORMATO DE REPRESENTAÇÃO DA SAÍDA DO ALGORITMO
    def __repr__(self):
        return f'<FoodSource src:{self.solution} fitness:{self.fitness} />'

    #ALGORITMO
class ABC(object):
    food_sources = []

    #DEFINIÇÕES DOS CAMPOS
    def __init__(self, n_pop, n_runs, fn_eval, *, trials_limit=100, employed=0.5, fn_lb=[], fn_ub=[], ):
        super(ABC, self).__init__()
        self.n_population = n_pop
        self.n_runs = n_runs
        self.fn_eval = fn_eval
        self.trials_limit = trials_limit
        self.fn_lb = np.array(fn_lb)
        self.fn_ub = np.array(fn_ub)
        self.employed_bees = round(n_pop * employed)
        self.onlooker_bees = n_pop - self.employed_bees
        self.history_fitness = []

    #PARTE PRINCIPAL
    def optimize(self):
        self.initialize()

        for n_run in range(1, self.n_runs + 1):
            self.employed()
            self.onlooker()
            self.scout()

        best_fs = self.best_source()
        self.plot_graph(self.history_fitness)

        return best_fs.solution

    def initialize(self):
        self.food_sources = [self.create_foodsource() for i in range(self.employed_bees)]

    #TIPOS DOS 3 TIPOS DE AGENTES(ABELHAS)
    def employed(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]
            new_solution = self.generate_solution(i)
            #self.history_fitness.append(new_solution)
            best_solution = self.best_solution(food_source.solution, new_solution)

            self.set_solution(food_source, best_solution)

    def onlooker(self):
        for i in range(self.onlooker_bees):
            probabilities = [self.probability(fs) for fs in self.food_sources]
            selected_index = self.selection(range(len(self.food_sources)), probabilities)
            selected_source = self.food_sources[selected_index]
            new_solution = self.generate_solution(selected_index)
            best_solution = self.best_solution(selected_source.solution, new_solution)

            self.set_solution(selected_source, best_solution)

    def scout(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]

            if food_source.trials > self.trials_limit:
                food_source = self.create_foodsource()

    #CALCULO DA SOLUÇÃO
    def generate_solution(self, current_solution_index):
        solution = self.food_sources[current_solution_index].solution
        k_source_index = self.random_solution_excluding([current_solution_index])
        k_solution = self.food_sources[k_source_index].solution
        d = rand.randint(0, len(self.fn_lb) - 1)
        r = rand.uniform(-1, 1)

        new_solution = np.copy(solution)
        new_solution[d] = solution[d] + r * (solution[d] - k_solution[d])
        return np.around(new_solution, decimals=6)

    def random_solution_excluding(self, excluded_index):
        available_indexes = set(range(self.employed_bees))
        exclude_set = set(excluded_index)
        diff = available_indexes - exclude_set
        selected = rand.choice(list(diff))
        return selected

    def best_solution(self, current_solution, new_solution):
        if self.fitness(new_solution) > self.fitness(current_solution):
            return new_solution
        else:
            return current_solution

    def probability(self, solution_fitness):
        fitness_sum = sum([fs.fitness for fs in self.food_sources])
        probability = solution_fitness.fitness / fitness_sum
        return probability

    #FITNESS DA FOODSOURCE
    def fitness(self, solution):
        result = self.fn_eval(solution)

        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = abs(result)

        self.history_fitness.append(fitness)
        return fitness

    #seleção
    def selection(self, solutions, weights):
        return rand.choices(solutions, weights)[0]

    def set_solution(self, food_source, new_solution):
        if np.array_equal(new_solution, food_source.solution):
            food_source.trials += 1
        else:
            food_source.solution = new_solution
            food_source.trials = 0

    def best_source(self):
        best = max(self.food_sources, key=attrgetter('fitness'))
        return best

    #criação da foodsource
    def create_foodsource(self):
        solution = self.candidate_solution(self.fn_lb, self.fn_ub)
        fitness = self.fitness(solution)
        return FoodSource(solution, fitness)

    def candidate_solution(self, lb, ub):
        r = rand.random()
        solution = lb + (ub - lb) * r

        return np.around(solution, decimals=6)

    def plot_graph(self, history_fit):
        plt.scatter(range(len(history_fit)), history_fit, s=1.5)

        return plt.show()

#FUNÇÕES DE TESTE----
def sphere(d):
    return np.sum([x**2 for x in d])

def rastrigin(d):
    sum_i = np.sum([x**2 - 10*np.cos(2 * np.pi * x) for x in d])
    return 10 * len(d) + sum_i

def rosenbrock(d):
    x = d[0]
    y = d[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.


#MAIN----------------
def main():
        #limites do espaço de busca
        randLow = rand.random() *(-1)
        randPosLow = [randLow] *30
        randPosUp = [rand.random()] *30

        #formato: ABC(populacao, numero de repetiçoes, função a ser otimizada, limite mínimo no espaço de busca, limite máximo no espaço de busca)
        abc_sphere = ABC(30, 20, sphere, fn_lb=randPosLow, fn_ub=randPosUp)

        abc_rastrigin = ABC(30, 100, rastrigin, fn_lb=randPosLow, fn_ub=randPosUp)

        abc_rosenbrock = ABC(30, 50, rosenbrock, fn_lb=randPosLow, fn_ub=randPosUp)

#SELETOR FUNÇÕES A SEREM OTIMIZADAS
#Sphere
        show_result = abc_sphere.optimize()

#Rastrigin
        #show_result = abc_rastrigin.optimize()

#Rosenbrock
        #show_result = abc_rosenbrock.optimize()


if __name__ == '__main__':
    main()
