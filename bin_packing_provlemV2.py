import random
import copy
from collections import deque
import math


# Objective function
def objective_function(bins, bin_capacity, expected_items_count=None):
    total_items = sum(len(bin) for bin in bins)
    if expected_items_count is not None and total_items != expected_items_count:
        return float('inf')
    for bin in bins:
        if sum(bin) > bin_capacity:
            return float('inf')
    return len(bins)


# Random Solution
def random_solution(items, bin_capacity):
    bins = []
    random.shuffle(items)

    for item in items:
        valid_bins = [b for b in bins if sum(b) + item <= bin_capacity]

        if valid_bins:
            chosen_bin = random.choice(valid_bins)
            chosen_bin.append(item)
        else:
            bins.append([item])
    return bins


# Generowanie neighbours
def generate_neighbours_by_permutation(items):
    neighbours = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            neighbour = items.copy()
            neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
            neighbours.append(neighbour)
    return neighbours


def naive_pack(items, bin_capacity):
    bins = []
    for item in items:
        for b in bins:
            if sum(b) + item <= bin_capacity:
                b.append(item)
                break
        else:
            bins.append([item])
    return bins


# Algorytm pelnego przegladu
def generate_all_valid_bin_packings(items, bin_capacity):
    results = []

    def backtrack(remaining_items, current_bins):
        if not remaining_items:
            results.append(current_bins)
            return

        for i in range(len(current_bins)):
            bin_copy = current_bins[i][:]
            if sum(bin_copy) + remaining_items[0] <= bin_capacity:
                bin_copy.append(remaining_items[0])
                new_bins = current_bins[:i] + [bin_copy] + current_bins[i+1:]
                backtrack(remaining_items[1:], new_bins)


        backtrack(remaining_items[1:], current_bins + [[remaining_items[0]]])

    # Tylko jedna permutacja (bo wszystkie rozkłady będą sprawdzane)
    backtrack(items, [])
    return results


def brute_force_optimal_bin_packing(items, bin_capacity):
    best_solution = None
    best_cost = float('inf')

    # Sprawdzamy tylko jedną permutację, ale rozdzielamy wszystkie możliwe biny
    all_packings = generate_all_valid_bin_packings(items, bin_capacity)

    for packing in all_packings:
        cost = objective_function(packing, bin_capacity)
        if cost < best_cost:
            best_cost = cost
            best_solution = packing

    return best_solution, best_cost

# Hill Climbing best neighbour
def hill_climbing_bin_packing(items, bin_capacity, max_iterations):
    current_items = items.copy()
    random.shuffle(current_items)

    current_bins = naive_pack(current_items, bin_capacity)
    current_cost = objective_function(current_bins, bin_capacity)

    for _ in range(max_iterations):
        neighbours = generate_neighbours_by_permutation(current_items)
        improved = False

        for neighbour_items in neighbours:
            neighbour_bins = naive_pack(neighbour_items, bin_capacity)
            neighbour_cost = objective_function(neighbour_bins, bin_capacity)

            if neighbour_cost < current_cost:
                current_items = neighbour_items
                current_bins = neighbour_bins
                current_cost = neighbour_cost
                improved = True
                break

        if not improved:
            break

    return current_bins, current_cost


# Hill Climb random
def stochastic_hill_climbing_bin_packing(items, bin_capacity, max_iterations=1000):
    current_items = items.copy()
    random.shuffle(current_items)

    current_bins = naive_pack(current_items, bin_capacity)
    current_cost = objective_function(current_bins, bin_capacity)

    for _ in range(max_iterations):
        neighbours = generate_neighbours_by_permutation(current_items)
        better_neighbours = []

        for neighbour_items in neighbours:
            neighbour_bins = naive_pack(neighbour_items, bin_capacity)
            neighbour_cost = objective_function(neighbour_bins, bin_capacity)

            if neighbour_cost < current_cost:
                better_neighbours.append((neighbour_items, neighbour_bins, neighbour_cost))

        if not better_neighbours:
            break

        # losowy z lepszych
        selected = random.choice(better_neighbours)
        current_items, current_bins, current_cost = selected

    return current_bins, current_cost



# Algorytm tabu search
def tabu_search_bin_packing(items, bin_capacity, tabu_size=10, max_iterations=1000):
    current_items = items.copy()
    random.shuffle(current_items)

    current_bins = naive_pack(current_items, bin_capacity)
    current_cost = objective_function(current_bins, bin_capacity)

    best_items = current_items.copy()
    best_bins = current_bins
    best_cost = current_cost

    tabu_list = deque(maxlen=tabu_size)
    backup_stack = []

    for _ in range(max_iterations):
        neighbours = generate_neighbours_by_permutation(current_items)

        best_candidate = None
        best_candidate_bins = None
        best_candidate_cost = float('inf')

        for neighbour_items in neighbours:
            key = tuple(neighbour_items)
            if key in tabu_list and cost >= best_cost:
                continue  # odrzucamy tylko, jeśli jest w tabu i nie poprawia

            bins = naive_pack(neighbour_items, bin_capacity)
            cost = objective_function(bins, bin_capacity)

            if cost < best_candidate_cost:
                best_candidate = neighbour_items
                best_candidate_bins = bins
                best_candidate_cost = cost

        if best_candidate is None:
            if backup_stack:
                current_items = backup_stack.pop()
                continue
            else:
                break

        backup_stack.append(current_items.copy())
        current_items = best_candidate
        current_bins = best_candidate_bins
        current_cost = best_candidate_cost
        tabu_list.append(tuple(current_items))

        if current_cost < best_cost:
            best_items = current_items.copy()
            best_bins = current_bins
            best_cost = current_cost

    return best_bins, best_cost

#Wyzazanie
def simulated_annealing_bin_packing(items, bin_capacity, T0=100.0, max_iterations=1000,
                                     cooling_schedule=None):
    current_items = items.copy()
    random.shuffle(current_items)
    current_bins = naive_pack(current_items, bin_capacity)
    current_cost = objective_function(current_bins, bin_capacity)

    best_items = current_items.copy()
    best_bins = current_bins
    best_cost = current_cost

    for k in range(1, max_iterations + 1):
        # Generowanie losowego sąsiada
        i, j = random.sample(range(len(current_items)), 2)
        neighbour_items = current_items.copy()
        neighbour_items[i], neighbour_items[j] = neighbour_items[j], neighbour_items[i]

        neighbour_bins = naive_pack(neighbour_items, bin_capacity)
        neighbour_cost = objective_function(neighbour_bins, bin_capacity)

        delta = neighbour_cost - current_cost
        temperature = cooling_schedule(k) if cooling_schedule else 1.0

        # Akceptacja warunkowa
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_items = neighbour_items
            current_bins = neighbour_bins
            current_cost = neighbour_cost

            if current_cost < best_cost:
                best_items = current_items.copy()
                best_bins = current_bins
                best_cost = current_cost

    return best_bins, best_cost

def linear_cooling(T0, delta):
    return lambda k: max(T0 - k * delta, 0.001)

def exponential_cooling(T0=100.0, alpha=0.95):
    return lambda k: T0 * (alpha ** k)



#===========Genetyka====================

# Krzyżowanie 1-punktowe
def crossover_one_point(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child = parent1[:point] + [item for item in parent2 if item not in parent1[:point]]
    return child

# Krzyżowanie typu Order Crossover (OX)
def crossover_order(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    middle = parent1[start:end]
    rest = [item for item in parent2 if item not in middle]
    child = rest[:start] + middle + rest[start:]
    return child

# Mutacja przez zamianę dwóch genów
def mutation_swap(individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

# Mutacja przez odwrócenie fragmentu
def mutation_reverse_segment(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))
    individual[start:end] = reversed(individual[start:end])
    return individual

def termination_max_generations(current_generation, max_generations, **kwargs):
    return current_generation >= max_generations

def termination_no_improvement(current_generation, max_generations, best_cost, last_improvement_gen, patience=20):
    return (current_generation - last_improvement_gen) >= patience or current_generation >= max_generations

def genetic_algorithm_bin_packing(items, bin_capacity,
                                  population_size=50,
                                  crossover_method='one_point',
                                  mutation_method='swap',
                                  termination='generations',
                                  max_generations=100,
                                  use_elitism=True):

    crossover_functions = {
        'one_point': crossover_one_point,
        'order': crossover_order
    }
    mutation_functions = {
        'swap': mutation_swap,
        'reverse': mutation_reverse_segment
    }
    termination_functions = {
        'generations': termination_max_generations,
        'no_improvement': termination_no_improvement
    }

    crossover = crossover_functions[crossover_method]
    mutate = mutation_functions[mutation_method]
    should_terminate = termination_functions[termination]

    # Inicjalizacja populacji
    population = [random.sample(items, len(items)) for _ in range(population_size)]
    expected_items_count = len(items)
    population_fitness = [objective_function(naive_pack(ind, bin_capacity), bin_capacity, expected_items_count) for ind
                          in population]

    best_solution = population[population_fitness.index(min(population_fitness))]
    best_cost = min(population_fitness)
    last_improvement_gen = 0

    for generation in range(max_generations):
        # Selekcja rankingowa – sortowanie po przystosowaniu
        sorted_population = [x for _, x in sorted(zip(population_fitness, population))]
        next_population = []

        # Elita
        if use_elitism:
            next_population.append(best_solution)

        # Tworzenie nowego pokolenia
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(sorted_population[:population_size // 2], 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # Prawdopodobieństwo mutacji
                child = mutate(child)
            next_population.append(child)

        # Aktualizacja populacji
        population = next_population
        expected_items_count = len(items)
        population_fitness = [objective_function(naive_pack(ind, bin_capacity), bin_capacity, expected_items_count) for
                              ind in population]

        current_best_cost = min(population_fitness)
        current_best_solution = population[population_fitness.index(current_best_cost)]

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = current_best_solution
            last_improvement_gen = generation

        # Sprawdzenie zakończenia
        if termination == 'generations':
            if should_terminate(generation, max_generations=max_generations):
                break
        elif termination == 'no_improvement':
            if should_terminate(generation, max_generations=max_generations,
                                best_cost=best_cost, last_improvement_gen=last_improvement_gen):
                break

    return naive_pack(best_solution, bin_capacity), best_cost


def main(items, bin_capacity, max_iterations=1000):
    print("=== Bin Packing Problem ===")
    print(f"Items: {items}")
    print(f"Bin capacity: {bin_capacity}")
    print("---------------------------")

    # Random Solution
    bins_random = random_solution(items, bin_capacity)
    cost_random = objective_function(bins_random, bin_capacity)
    print(f"Random Solution | Liczba pojemników: {len(bins_random)}, Koszt: {cost_random}")
    print(bins_random)
    print("---------------------------")

    # Hill Climbing (Best Neighbour)
    bins_hc_best, cost_hc_best = hill_climbing_bin_packing(items, bin_capacity, max_iterations)
    print(f"Hill Climbing (Best Neighbour) | Liczba pojemników: {len(bins_hc_best)}, Koszt: {cost_hc_best}")
    print(bins_hc_best)
    print("---------------------------")

    # Hill Climbing (Stochastic)
    bins_hc_random, cost_hc_random = stochastic_hill_climbing_bin_packing(items, bin_capacity, max_iterations)
    print(f"Hill Climbing (Random Neighbour) | Liczba pojemników: {len(bins_hc_random)}, Koszt: {cost_hc_random}")
    print(bins_hc_random)
    print("---------------------------")

    # Tabu Search
    bins_tabu, cost_tabu = tabu_search_bin_packing(items, bin_capacity, tabu_size=5, max_iterations=max_iterations)
    print(f"Tabu Search | Liczba pojemników: {len(bins_tabu)}, Koszt: {cost_tabu}")
    print(bins_tabu)
    print("---------------------------")

    # Brute Force
    if len(items) <= 10:
        bins_brute, cost_brute = brute_force_optimal_bin_packing(items, bin_capacity)
        print(f"Brute Force (Poprawny) | Liczba pojemników: {len(bins_brute)}, Koszt: {cost_brute}")
        print(bins_brute)
    else:
        print("Brute Force pominięty (problem zbyt duży)")


    cooling = exponential_cooling(T0=100.0, alpha=0.95)
    bins_sa, cost_sa = simulated_annealing_bin_packing(items, bin_capacity,
                                                       T0=100.0,
                                                       max_iterations=max_iterations,
                                                       cooling_schedule=cooling)
    print(f"Simulated Annealing | Liczba pojemników: {len(bins_sa)}, Koszt: {cost_sa}")
    print(bins_sa)
    print("---------------------------")

    # Genetic Algorithm
    bins_ga, cost_ga = genetic_algorithm_bin_packing(
        items,
        bin_capacity,
        population_size=50,
        crossover_method='order',
        mutation_method='reverse',
        termination='no_improvement',
        max_generations=max_iterations,
        use_elitism=True
    )
    print(f"Genetic Algorithm | Liczba pojemników: {len(bins_ga)}, Koszt: {cost_ga}")
    print(bins_ga)
    print("---------------------------")


# Uruchomienie testu z dużym zestawem elementów
if __name__ == '__main__':
    items = [3, 7, 4, 6, 5, 5, 5, 5, 5, 5] * 5
    bin_capacity = 10
    max_iterations = 1000

    main(items, bin_capacity, max_iterations)
