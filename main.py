import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsplib95
import csv

# Settings
SEED = 13
ITERATIONS = 1000
RUNS = 10
# DATASETS = [
#     "berlin52",
#     "kroA100"]
DATASETS = ["datasets/Delivery_Logistics.csv"]
CAPACITY = 1000

# Loads tsplib datasets. Will probably be deprecated later
def load_problem_tsp(name):
    problem = tsplib95.load(f"datasets/{name}.tsp")
    cities = list(problem.get_nodes())
    distance_matrix = {
        (i, j): problem.get_weight(i, j)
        for i in cities
        for j in cities}

    return problem, cities, distance_matrix

def load_problem_knapsack(file_path):
    mode_mapping = {"standard": 1, "two day": 2, "same day": 3, "express": 4}
    df = pd.read_csv(file_path)
    result = df[["package_weight_kg", "delivery_mode"]].copy()
    result["delivery_mode"] = (result["delivery_mode"].astype(str).str.strip().str.lower().map(mode_mapping))
    result = result.rename(columns={"package_weight_kg": "weight", "delivery_mode": "value"})
    result["value"] = result["value"].astype(int)
    result = result.head(300).reset_index(drop=True)
    weights = result["weight"].to_numpy()
    values = result["value"].to_numpy()
    return weights, values

# For calculating the distance
def route_distance(route, distance_matrix):
    total = 0
    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]
        total += distance_matrix[(current_city, next_city)]
    
    return total
# Add penalty if infeasible
def knapsack_value(solution, weights, values, capacity):
    total_weight = sum(bit * weight for bit, weight in zip(solution, weights))
    if total_weight > capacity:
        return 0
    
    total_value = sum(bit * value for bit, value in zip(solution, values))
    return total_value

# Starting random route
def random_route(cities):
    route = cities[:]
    random.shuffle(route)
    return route

# Greedy route.
def greedy_route(cities, distance_matrix):
    unvisited = set(cities)
    start = random.choice(cities)
    route = [start]
    unvisited.remove(start)

    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda c: distance_matrix[(current, c)])
        route.append(nearest)
        unvisited.remove(nearest)

    return route

def knapsack_random_solution(weights, capacity):
    """Generate a random feasible solution, not necessarily trying to fill capacity"""
    n_items = len(weights)
    solution = [0] * n_items
    indices = list(range(n_items))
    random.shuffle(indices)
    current_weight = 0
    
    # Add items randomly until we have at least 1 item or can't add more
    for idx in indices:
        if current_weight + weights[idx] <= capacity:
            solution[idx] = 1
            current_weight += weights[idx]
            # Stop after adding a random number of items (1-5) for more exploration
            if random.random() < 0.3:  # 30% chance to stop early
                break
    print(solution)
    print(current_weight)
    return solution

# Swap two nodes
def swap(route):
    new_route = route[:]
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# Remove and reinsert node
def insertion(route):
    new_route = route[:]
    i, j = random.sample(range(len(route)), 2)
    city = new_route.pop(i)
    new_route.insert(j, city)
    return new_route

# Segment shuffle
def segment_shuffle(route):
    new_route = route[:]
    i, j = sorted(random.sample(range(len(route)), 2))
    # Ensure minimum segment size of at least 5 for meaningful shuffling
    if j - i < 5:
        j = min(i + random.randint(5, max(6, len(route) // 3)), len(route))
    segment = new_route[i:j]
    random.shuffle(segment)
    new_route[i:j] = segment
    return new_route

def two_opt(route):
    new_route = route[:]
    i, j = sorted(random.sample(range(len(route)), 2))
    if j - i < 4:
        j = i + 4  # Ensure at least 2 cities are swapped
    new_route[i:j] = list(reversed(new_route[i:j]))
    return new_route

tsp_heuristics = [
    ("swap", swap),
    ("insertion", insertion),
    ("segment_shuffle", segment_shuffle),
    ("two opt", two_opt),]

def flip_k_bits(solution, k):
    new_solution = solution.copy()
    indices = random.sample(range(len(solution)), k)
    for idx in indices:
        new_solution[idx] = 1 - new_solution[idx]
    return new_solution


def one_bit_flip(solution):
    return flip_k_bits(solution, 1)

def two_bit_flip(solution):
    return flip_k_bits(solution, 2)

def three_bit_flip(solution):
    return flip_k_bits(solution, 3)

def four_bit_flip(solution):
    return flip_k_bits(solution, 4)
# Add something that has a chance to flip all bits. 1/prob
knapsack_heuristics = [
    ("one_bit_flip", one_bit_flip),
    ("two_bit_flip", two_bit_flip),
    ("three_bit_flip", three_bit_flip),
    ("four_bit_flip", four_bit_flip),]

# For keeping track of which heuristics were used
def initialize_stats(heuristics = tsp_heuristics):
    return {name: {
            "used": 0,
            "improvements": 0,
            "total_gain": 0,
            "best_single_gain": 0}
        for name, _ in heuristics}


def random_hyperheuristic(initial_solution, objective_fn, heuristics, iterations=25000):
    # minimises the objective function.
    # For maximisation problems should use a negative function.

    current = initial_solution[:]
    current_cost = objective_fn(current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    start_time = time.time()

    for _ in range(iterations):

        heuristic_name, heuristic = random.choice(heuristics)
        heuristic_history.append(heuristic_name)
        heuristic_stats[heuristic_name]["used"] += 1
        candidate = heuristic(current)
        candidate_cost = objective_fn(candidate)

        # Accept only improving solutions
        if candidate_cost < current_cost:

            gain = current_cost - candidate_cost

            current = candidate
            current_cost = candidate_cost

            heuristic_stats[heuristic_name]["improvements"] += 1
            heuristic_stats[heuristic_name]["total_gain"] += gain

            if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                heuristic_stats[heuristic_name]["best_single_gain"] = gain

        if current_cost < best_cost:

            best = current[:]
            best_cost = current_cost

        history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "runtime": runtime}


def random_gradient(initial_solution, objective_fn, heuristics, iterations=25000):
    # objective_fn(solution) is always minimised.
    # For maximisation problems like knapsack, use a negative objective.

    current = initial_solution[:]
    current_cost = objective_fn(current)

    best = current[:]
    best_cost = current_cost

    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    heuristic_name, heuristic = random.choice(heuristics)
    start_time = time.time()

    for _ in range(iterations):

        heuristic_history.append(heuristic_name)
        heuristic_stats[heuristic_name]["used"] += 1
        candidate = heuristic(current)
        candidate_cost = objective_fn(candidate)

        # Improvement found: keep heuristic
        if candidate_cost < current_cost:

            gain = current_cost - candidate_cost
            current = candidate
            current_cost = candidate_cost
            heuristic_stats[heuristic_name]["improvements"] += 1
            heuristic_stats[heuristic_name]["total_gain"] += gain

            if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                heuristic_stats[heuristic_name]["best_single_gain"] = gain

        # No improvement: switch heuristic
        else:
            other_heuristics = [(n, h) for n, h in heuristics if n != heuristic_name]
            heuristic_name, heuristic = random.choice(other_heuristics)

        if current_cost < best_cost:

            best = current[:]
            best_cost = current_cost

        history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "runtime": runtime}


def random_permutation(initial_solution, objective_fn, heuristics, iterations=25000):
    # At each step, applies heuristics in a new shuffled order.

    current = initial_solution[:]
    current_cost = objective_fn(current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    start_time = time.time()

    permutation = list(heuristics)
    random.shuffle(permutation)
    perm_index = 0

    for _ in range(iterations):

        if perm_index >= len(permutation):
            random.shuffle(permutation)
            perm_index = 0

        heuristic_name, heuristic = permutation[perm_index]
        perm_index += 1

        heuristic_history.append(heuristic_name)
        heuristic_stats[heuristic_name]["used"] += 1
        candidate = heuristic(current)
        candidate_cost = objective_fn(candidate)

        # Accept only improving solutions
        if candidate_cost < current_cost:

            gain = current_cost - candidate_cost
            current = candidate
            current_cost = candidate_cost
            heuristic_stats[heuristic_name]["improvements"] += 1
            heuristic_stats[heuristic_name]["total_gain"] += gain

            if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                heuristic_stats[heuristic_name]["best_single_gain"] = gain

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "runtime": runtime}

# Creates a random heuristic list, but keeps applying the current heuristic as long as it keeps improving.
def permutation_gradient(initial_solution, objective_fn, heuristics, iterations=25000):
    current = initial_solution[:]
    current_cost = objective_fn(current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    start_time = time.time()

    permutation = list(heuristics)
    random.shuffle(permutation)
    perm_index = 0
    heuristic_name, heuristic = permutation[perm_index]

    for _ in range(iterations):

        heuristic_history.append(heuristic_name)
        heuristic_stats[heuristic_name]["used"] += 1
        candidate = heuristic(current)
        candidate_cost = objective_fn(candidate)

        # Improvement found: keep heuristic
        if candidate_cost < current_cost:

            gain = current_cost - candidate_cost
            current = candidate
            current_cost = candidate_cost
            heuristic_stats[heuristic_name]["improvements"] += 1
            heuristic_stats[heuristic_name]["total_gain"] += gain

            if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                heuristic_stats[heuristic_name]["best_single_gain"] = gain

        # No improvement
        else:

            perm_index += 1
            if perm_index >= len(permutation):
                random.shuffle(permutation)
                perm_index = 0

            heuristic_name, heuristic = permutation[perm_index]

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "runtime": runtime}

# Each iteration applies all heuristics to the current solution and accepts the one that gives the best improvement.
def greedy(initial_solution, objective_fn, heuristics, iterations=25000):
    current = initial_solution[:]
    current_cost = objective_fn(current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    start_time = time.time()
    iteration = 0

    while iteration < iterations:

        best_candidate = None
        best_candidate_cost = current_cost
        best_candidate_name = None

        for heuristic_name, heuristic in heuristics:

            if iteration >= iterations:
                break

            heuristic_stats[heuristic_name]["used"] += 1
            heuristic_history.append(heuristic_name)
            candidate = heuristic(current)
            candidate_cost = objective_fn(candidate)
            iteration += 1

            if candidate_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = candidate_cost
                best_candidate_name = heuristic_name

        # Accept the best candidate from this round if it improves
        if best_candidate is not None:

            gain = current_cost - best_candidate_cost
            current = best_candidate
            current_cost = best_candidate_cost
            heuristic_stats[best_candidate_name]["improvements"] += 1
            heuristic_stats[best_candidate_name]["total_gain"] += gain

            if gain > heuristic_stats[best_candidate_name]["best_single_gain"]:
                heuristic_stats[best_candidate_name]["best_single_gain"] = gain

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "runtime": runtime}


def greedy_gradient(initial_solution, objective_fn, heuristics, iterations=25000):
    # Runs a greedy round to find the best heuristic, then keeps applying it as long as it improves.

    current = initial_solution[:]
    current_cost = objective_fn(current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    start_time = time.time()
    iteration = 0

    def greedy_round():
        # Tries all heuristics once, returns the name/fn with the best gain (or None).
        best_candidate = None
        best_candidate_cost = current_cost
        best_fn = None
        best_name = None

        for heuristic_name, heuristic in heuristics:
            heuristic_stats[heuristic_name]["used"] += 1
            heuristic_history.append(heuristic_name)
            candidate = heuristic(current)
            candidate_cost = objective_fn(candidate)

            if candidate_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = candidate_cost
                best_fn = heuristic
                best_name = heuristic_name

        return best_name, best_fn, best_candidate, best_candidate_cost

    # Initial greedy round to pick the starting heuristic
    iteration += len(heuristics)
    heuristic_name, heuristic, first_candidate, first_cost = greedy_round()
    history.append(best_cost)

    # Accept the initial greedy winner if it improved
    if heuristic is not None:

        gain = current_cost - first_cost
        current = first_candidate
        current_cost = first_cost
        heuristic_stats[heuristic_name]["improvements"] += 1
        heuristic_stats[heuristic_name]["total_gain"] += gain

        if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
            heuristic_stats[heuristic_name]["best_single_gain"] = gain

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

    while iteration < iterations:

        if heuristic is None:
            # No heuristic improved last round — run a new greedy round
            iteration += len(heuristics)
            heuristic_name, heuristic, candidate, candidate_cost = greedy_round()
            history.append(best_cost)

            if heuristic is not None:
                gain = current_cost - candidate_cost
                current = candidate
                current_cost = candidate_cost
                heuristic_stats[heuristic_name]["improvements"] += 1
                heuristic_stats[heuristic_name]["total_gain"] += gain

                if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                    heuristic_stats[heuristic_name]["best_single_gain"] = gain

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

            continue

        # Keep applying the winning heuristic
        heuristic_history.append(heuristic_name)
        heuristic_stats[heuristic_name]["used"] += 1
        candidate = heuristic(current)
        candidate_cost = objective_fn(candidate)
        iteration += 1

        if candidate_cost < current_cost:

            gain = current_cost - candidate_cost
            current = candidate
            current_cost = candidate_cost
            heuristic_stats[heuristic_name]["improvements"] += 1
            heuristic_stats[heuristic_name]["total_gain"] += gain

            if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                heuristic_stats[heuristic_name]["best_single_gain"] = gain

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        else:
            # Current heuristic failed — run a new greedy round to re-select
            iteration += len(heuristics)
            heuristic_name, heuristic, candidate, candidate_cost = greedy_round()
            history.append(best_cost)

            if heuristic is not None:
                gain = current_cost - candidate_cost
                current = candidate
                current_cost = candidate_cost
                heuristic_stats[heuristic_name]["improvements"] += 1
                heuristic_stats[heuristic_name]["total_gain"] += gain

                if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                    heuristic_stats[heuristic_name]["best_single_gain"] = gain

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

        history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "runtime": runtime}

if __name__ == "__main__":
    all_results = []

    for dataset_name in DATASETS:

        print(f"Dataset: {dataset_name}")
        algorithms = [
            ("Random HH",            random_hyperheuristic),
            ("RandomGradient HH",    random_gradient),
            ("RandomPermutation HH", random_permutation),
            ("PermutationGradient HH", permutation_gradient),
            ("Greedy HH",            greedy),
            ("GreedyGradient HH",    greedy_gradient),]

        all_run_results = {label: [] for label, _ in algorithms}

        # TSP settings.
        # problem, cities, distance_matrix = load_problem(dataset_name)
        # def tsp_objective(route):
        #     return route_distance(route, distance_matrix)
        # initial = random_route(cities)
        # objective = tsp_objective
        # heuristics = tsp_heuristics

        # Knapsack settings
        weights, values = load_problem_knapsack(dataset_name)
        def knapsack_objective(solution):
            return -knapsack_value(solution, weights, values, CAPACITY)
        initial = knapsack_random_solution(weights, CAPACITY)
        objective = knapsack_objective
        heuristics = knapsack_heuristics

        for run in range(RUNS):
            print(f"Run {run + 1}/{RUNS}")

            for label, runner in algorithms:
                random.seed(SEED + run)
                np.random.seed(SEED + run)
                result = runner(initial, objective, heuristics, ITERATIONS)
                all_run_results[label].append(result)

        dataset_row = {"dataset": dataset_name}

        for label, _ in algorithms:
            costs = [r["best_cost"] for r in all_run_results[label]]
            slug = label.lower().replace(" ", "_")

            print(f"\n{label}")
            print("Average:", np.mean(costs))
            print("Std:    ", np.std(costs))
            print("Best:   ", np.min(costs))

            dataset_row[f"{slug}_avg"]  = np.mean(costs)
            dataset_row[f"{slug}_std"]  = np.std(costs)
            dataset_row[f"{slug}_best"] = np.min(costs)

        all_results.append(dataset_row)

        plt.figure(figsize=(12, 6))

        for label, _ in algorithms:
            best_result = min(all_run_results[label], key=lambda r: r["best_cost"])
            plt.plot(best_result["history"], label=label)

        plt.xlabel("Iteration")
        plt.ylabel("Best Value")
        plt.title(f"{dataset_name} — Convergence (best run per algorithm)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_convergence.png", dpi=150, bbox_inches="tight")
        plt.close()

        for label, _ in algorithms:

            best_result = min(all_run_results[label], key=lambda r: r["best_cost"])
            stats = best_result["heuristic_stats"]
            print(f"\n--- {label} heuristic stats (best run) ---")

            for heuristic_name, values in stats.items():

                used = values["used"]
                improvements = values["improvements"]
                acceptance_rate = improvements / used if used > 0 else 0
                print(f"\n  {heuristic_name}")
                print("Used:", used)
                print("Improvements:", improvements)
                print("Acceptance rate:", round(acceptance_rate, 4))
                print("Total gain:", round(values["total_gain"], 2))
                print("Best single gain:", round(values["best_single_gain"], 2))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results_summary.csv", index=False)