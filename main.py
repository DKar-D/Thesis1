import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsplib95

# Settings. There are also some settings at line 728
SEED = 13
ITERATIONS = 20000
RUNS = 20
# DATASETS = ["datasets/Delivery_Logistics.csv"]
DATASETS = ["500", "berlin52",]
CAPACITY = 3000
KNAPSACK_ITEMS = 2500
TAU = 150
# PROBLEM = "Knapsack"
PROBLEM = "TSP"

# Loads tsplib datasets.
def load_problem_tsp(name):
    problem = tsplib95.load(f"datasets/{name}.tsp")
    cities = list(problem.get_nodes())
    distance_matrix = {
        (i, j): problem.get_weight(i, j)
        for i in cities
        for j in cities}

    return problem, cities, distance_matrix

def load_problem_knapsack(file_path, knapsack_items = KNAPSACK_ITEMS):
    mode_mapping = {"standard": 1, "two day": 2, "same day": 3, "express": 4}
    df = pd.read_csv(file_path)
    result = df[["package_weight_kg", "delivery_mode"]].copy()
    result["delivery_mode"] = (result["delivery_mode"].astype(str).str.strip().str.lower().map(mode_mapping))
    result = result.rename(columns={"package_weight_kg": "weight", "delivery_mode": "value"})
    result["value"] = result["value"].astype(int)
    result = result.head(knapsack_items).reset_index(drop=True)
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

def knapsack_simple_value(solution, weights, values, capacity):
    total_weight = sum(bit * weight for bit, weight in zip(solution, weights))
    if total_weight > capacity:
        return 0
    
    total_value = sum(bit * value for bit, value in zip(solution, values))
    return total_value

def knapsack_pen_value(solution, weights, values, capacity):
    total_weight = sum(bit * weight for bit, weight in zip(solution, weights))
    penalty = 0.4
    excess = 0
    if total_weight > capacity:
        excess = total_weight - capacity 
    
    total_value = sum(bit * value for bit, value in zip(solution, values))
    result = total_value - excess * penalty
    # print(f"value before penalty: {total_value}; after:{result}; excess: {excess}")
    return result

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
    n_items = len(weights)
    solution = [0] * n_items
    indices = list(range(n_items))
    random.shuffle(indices)
    current_weight = 0
    
    # Add items randomly until reaches capacity
    for idx in indices:
        if random.random() < 1:  # chance to stop. This is to make an empty or partially empty solution
            break
        if current_weight + weights[idx] <= capacity:
            solution[idx] = 1
            current_weight += weights[idx]

    print(solution)
    # print(current_weight)
    return solution

def knapsack_greedy_solution(weights, values, capacity):
    n_items = len(weights)
    solution = [0] * n_items

    # Sort item indices by value/weight ratio, highest first
    indices = sorted(
        range(n_items),
        key=lambda i: values[i] / weights[i],
        reverse=True)

    current_weight = 0

    for idx in indices:
        if current_weight + weights[idx] <= capacity:
            solution[idx] = 1
            current_weight += weights[idx]
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
    # Ensure minimum segment size of at least n for meaningful shuffling
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
        j = i + 4  # Ensure at least n cities are swapped
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

def flip_many(solution, chance=0.005):
    new_solution = solution.copy()
    for i in range(len(new_solution)):
        if random.random() < chance:
            new_solution[i] = 1 - new_solution[i]
    return new_solution

def add_random_item(solution):
    new = solution.copy()
    selected = [i for i, b in enumerate(new) if not b]

    if selected:
        new[random.choice(selected)] = 1

    return new

def remove_random_item(solution):
    new = solution.copy()
    selected = [i for i, b in enumerate(new) if b]

    if selected:
        new[random.choice(selected)] = 0

    return new

def add_best_ratio(solution):
    new = solution.copy()
    current_weight = sum(w for b, w in zip(new, weights) if b)
    best = None
    best_ratio = -1

    for i in range(len(new)):
        if new[i]:
            continue

        if current_weight + weights[i] > CAPACITY:
            continue

        ratio = values[i] / weights[i]

        if ratio > best_ratio:
            best_ratio = ratio
            best = i

    if best is not None:
        new[best] = 1

    return new

def remove_worst_ratio(solution):
    new = solution.copy()
    selected = [i for i, b in enumerate(new) if b]

    if not selected:
        return new

    worst = min(selected, key=lambda i: values[i] / weights[i])
    new[worst] = 0
    return new


knapsack_heuristics = [
    ("one_bit_flip", one_bit_flip),
    ("two_bit_flip", two_bit_flip),
    ("three_bit_flip", three_bit_flip),
    ("four_bit_flip", four_bit_flip),
    ("flip_many", flip_many),
    # ("add_random_item", add_random_item),
    # ("remove_random_item", remove_random_item),
    # ("add_best_ratio", add_best_ratio),
    # ("remove_worst_ratio", remove_worst_ratio)
    ]

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


def random_gradient_tau(initial_solution, objective_fn, heuristics, iterations=25000, tau = TAU):
    current = initial_solution[:]
    current_cost = objective_fn(current)

    best = current[:]
    best_cost = current_cost
    
    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    heuristic_name, heuristic = random.choice(heuristics)
    start_time = time.time()
    count = tau
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
            count = tau
            # print(f"Improvement found. Tau = {count}")
            if gain > heuristic_stats[heuristic_name]["best_single_gain"]:
                heuristic_stats[heuristic_name]["best_single_gain"] = gain

        # No improvement, but still within learning rate
        elif count > 0: 
            
            # Mistake here. Wasted time and made the heurstic stats inaccurate, but shouldn't have affected the number of iterations or the final result

            # heuristic_history.append(heuristic_name)
            # heuristic_stats[heuristic_name]["used"] += 1
            # candidate = heuristic(current)
            # candidate_cost = objective_fn(candidate)
            count -= 1
            # print(f"No Improvement. Tau = {count}")
        # No improvement, learning rate over
        else:
            other_heuristics = [(n, h) for n, h in heuristics if n != heuristic_name]
            heuristic_name, heuristic = random.choice(other_heuristics)
            count = tau
            # print(f"Reset. Tau = {count}")

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


def adaptive_random_gradient(initial_solution, objective_fn, heuristics, iterations=25000, tau0=1.0, F=1.5, sigma=4):
    current = initial_solution[:]
    current_cost = objective_fn(current)

    best = current[:]
    best_cost = current_cost

    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    tau_history = []

    tau = tau0
    ct = 0   # iterations used in the current learning period
    cs = 0   # improvements found in the current learning period
    heuristic_name, heuristic = random.choice(heuristics)
    start_time = time.time()

    for _ in range(iterations):

        heuristic_history.append(heuristic_name)
        tau_history.append(tau)
        heuristic_stats[heuristic_name]["used"] += 1

        candidate = heuristic(current)
        candidate_cost = objective_fn(candidate)
        ct += 1

        if candidate_cost < current_cost:
            gain = current_cost - candidate_cost
            current = candidate
            current_cost = candidate_cost
            cs += 1

            stats = heuristic_stats[heuristic_name]
            stats["improvements"] += 1
            stats["total_gain"] += gain
            if gain > stats["best_single_gain"]:
                stats["best_single_gain"] = gain

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

            # Sigma improvements reached -> keep heuristic, shrink tau
            if cs >= sigma:
                cs = 0
                ct = 0
                tau = max(1.0, tau * F ** (-1 / sigma ** 2))
                history.append(best_cost)
                continue

        # Learning period used up without sigma improvements -> grow tau, switch heuristic
        if ct >= tau:
            tau *= F ** (1 / sigma)
            ct = 0
            cs = 0
            other_heuristics = [(n, h) for n, h in heuristics if n != heuristic_name]
            heuristic_name, heuristic = random.choice(other_heuristics)

        history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "tau_history": tau_history,
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
        for i in heuristics:
            history.append(best_cost)

    runtime = time.time() - start_time

    return {"best_solution": best,
            "best_cost": best_cost,
            "history": history,
            "heuristic_stats": heuristic_stats,
            "heuristic_history": heuristic_history,
            "runtime": runtime}


def greedy_gradient(initial_solution, objective_fn, heuristics, iterations=25000):
    # Tries all heuristics each round and keeps applying whichever improved most
    # until it stops improving, then runs another greedy round to re-select.
    # One history entry is recorded per round
    # and per single-heuristic exploitation step.

    current = initial_solution[:]
    current_cost = objective_fn(current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    heuristic_stats = initialize_stats(heuristics)
    heuristic_history = []
    start_time = time.time()
    iteration = 0

    def _accept(name, candidate, candidate_cost):
        # Helper: update state when a candidate strictly improves current_cost.
        nonlocal current, current_cost, best, best_cost
        gain = current_cost - candidate_cost
        current = candidate
        current_cost = candidate_cost
        heuristic_stats[name]["improvements"] += 1
        heuristic_stats[name]["total_gain"] += gain
        if gain > heuristic_stats[name]["best_single_gain"]:
            heuristic_stats[name]["best_single_gain"] = gain
        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

    def _greedy_round():
        # Tries every heuristic once and returns the name/fn/candidate of the best,
        # or (None, None, None, current_cost) if nothing improved.
        best_name, best_fn, best_cand, best_cand_cost = None, None, None, current_cost
        for name, fn in heuristics:
            heuristic_stats[name]["used"] += 1
            heuristic_history.append(name)
            cand = fn(current)
            cand_cost = objective_fn(cand)
            if cand_cost < best_cand_cost:
                best_name, best_fn, best_cand, best_cand_cost = name, fn, cand, cand_cost
        return best_name, best_fn, best_cand, best_cand_cost

    # Start with a greedy round to pick the first heuristic to exploit.
    active_name, active_fn = None, None

    while iteration < iterations:

        if active_fn is None:
            # No active heuristic — run a greedy round to find one.
            iteration += len(heuristics)
            active_name, active_fn, cand, cand_cost = _greedy_round()
            if active_fn is not None:
                _accept(active_name, cand, cand_cost)
            for i in heuristics:
                history.append(best_cost)

        else:
            # Exploit the current winning heuristic.
            heuristic_history.append(active_name)
            heuristic_stats[active_name]["used"] += 1
            cand = active_fn(current)
            cand_cost = objective_fn(cand)
            iteration += 1

            if cand_cost < current_cost:
                _accept(active_name, cand, cand_cost)
            else:
                # Heuristic stalled: trigger a new greedy round next iteration.
                active_name, active_fn = None, None
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
        #HHs set
        algorithms = [
            # ("Random HH",            random_hyperheuristic),
            # ("RandomGradient HH",    random_gradient),
            # ("RandomPermutation HH", random_permutation),
            # ("PermutationGradient HH", permutation_gradient),
            # ("Greedy HH",            greedy),
            # ("GreedyGradient HH",    greedy_gradient),
            # ("Random Gradient LR", random_gradient_tau),
            ("Adaptive RD LR", adaptive_random_gradient),]

        all_run_results = {label: [] for label, _ in algorithms}
        random.seed(SEED)
        np.random.seed(SEED)

        # TSP settings.
        if PROBLEM == "TSP":
            problem, cities, distance_matrix = load_problem_tsp(dataset_name)
            def tsp_objective(route):
                return route_distance(route, distance_matrix)
            initial = random_route(cities)
            # initial = greedy_route(cities, distance_matrix)
            objective = tsp_objective
            heuristics = tsp_heuristics

        # Knapsack settings
        if PROBLEM == "Knapsack":
            weights, values = load_problem_knapsack(dataset_name)
            def knapsack_objective(solution):
                # return -knapsack_simple_value(solution, weights, values, CAPACITY)
                return -knapsack_pen_value(solution, weights, values, CAPACITY)
            initial = knapsack_random_solution(weights, CAPACITY)
            # initial = knapsack_greedy_solution(weights, values, CAPACITY)
            objective = knapsack_objective
            heuristics = knapsack_heuristics

        for run in range(RUNS):
            print(f"Run {run + 1}/{RUNS}")

            for runner_num, (label, runner) in enumerate(algorithms):
                random.seed(SEED + run + runner_num)
                np.random.seed(SEED + run + runner_num)
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
            # print(f"\n--- {label} heuristic stats (best run) ---")

            # for heuristic_name, values in stats.items():

            #     used = values["used"]
            #     improvements = values["improvements"]
            #     acceptance_rate = improvements / used if used > 0 else 0
            #     print(f"\n  {heuristic_name}")
            #     print("Used:", used)
            #     print("Improvements:", improvements)
            #     print("Acceptance rate:", round(acceptance_rate, 4))
            #     print("Total gain:", round(values["total_gain"], 2))
            #     print("Best single gain:", round(values["best_single_gain"], 2))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results_summary.csv", index=False)