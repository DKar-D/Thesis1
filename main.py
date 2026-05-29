import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsplib95

# Settings
SEED = 42
ITERATIONS = 25000
RUNS = 10
DATASETS = [
    "berlin52",
    "kroA100"]



# Loads tsplib datasets. Will probably be deprecated later
def load_problem(name):
    problem = tsplib95.load(f"datasets/{name}.tsp")
    cities = list(problem.get_nodes())
    distance_matrix = {
        (i, j): problem.get_weight(i, j)
        for i in cities
        for j in cities}

    return problem, cities, distance_matrix


# For calculating the distance
def route_distance(route, distance_matrix):
    total = 0
    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]
        total += distance_matrix[(current_city, next_city)]

    return total

# Starting random route
def random_route(cities):
    route = cities[:]
    random.shuffle(route)
    return route


# Swap two nodes
def swap(route):
    new_route = route[:]
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


# Reverse segment
def inversion(route):
    new_route = route[:]
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j] = reversed(new_route[i:j])
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
    segment = new_route[i:j]
    random.shuffle(segment)
    new_route[i:j] = segment
    return new_route


heuristics = [
    ("swap", swap),
    ("inversion", inversion),
    ("insertion", insertion),
    ("segment_shuffle", segment_shuffle),]


# For keeping track of which heuristics were used
def initialize_stats():
    return {name: {
            "used": 0,
            "improvements": 0,
            "total_gain": 0,
            "best_single_gain": 0}
        for name, _ in heuristics}


def random_hyperheuristic(initial_solution, objective_fn, heuristics, iterations=25000):
    # objective_fn(solution) -> float, always minimised.
    # For maximisation problems (e.g. knapsack), pass a negated objective.

    current = initial_solution[:]
    current_cost = objective_fn(current)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    heuristic_stats = initialize_stats()
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
    # For maximisation problems like knapsack, pass a negated objective.

    current = initial_solution[:]
    current_cost = objective_fn(current)

    best = current[:]
    best_cost = current_cost

    history = [best_cost]
    heuristic_stats = initialize_stats()
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


all_results = []

for dataset_name in DATASETS:

    print(f"Dataset: {dataset_name}")
    problem, cities, distance_matrix = load_problem(dataset_name)

    # TSP objective: minimise total route distance
    def tsp_objective(route):
        return route_distance(route, distance_matrix)

    random_results = []
    rg_results = []

    for run in range(RUNS):

        print(f"Run {run + 1}/{RUNS}")

        random.seed(SEED + run)
        np.random.seed(SEED + run)

        initial = random_route(cities)

        random_result = random_hyperheuristic(
            initial,
            tsp_objective,
            heuristics,
            ITERATIONS)

        random.seed(SEED + run)
        np.random.seed(SEED + run)

        rg_result = random_gradient(
            initial,
            tsp_objective,
            heuristics,
            ITERATIONS)

        random_results.append(random_result)
        rg_results.append(rg_result)

    random_costs = [r["best_cost"] for r in random_results]
    rg_costs = [r["best_cost"] for r in rg_results]

    print("\nRandom HH")
    print("Average:", np.mean(random_costs))
    print("Std:", np.std(random_costs))
    print("Best:", np.min(random_costs))

    print("\nRandomGradient HH")
    print("Average:", np.mean(rg_costs))
    print("Std:", np.std(rg_costs))
    print("Best:", np.min(rg_costs))

    all_results.append({
        "dataset": dataset_name,
        "random_avg": np.mean(random_costs),
        "random_std": np.std(random_costs),
        "random_best": np.min(random_costs),
        "rg_avg": np.mean(rg_costs),
        "rg_std": np.std(rg_costs),
        "rg_best": np.min(rg_costs)})

    best_random_result = min(random_results, key=lambda r: r["best_cost"])
    best_rg_result = min(rg_results, key=lambda r: r["best_cost"])

    plt.figure(figsize=(10, 5))

    plt.plot(
        best_random_result["history"],
        label="Random HH")

    plt.plot(
        best_rg_result["history"],
        label="RandomGradient HH")

    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.title(f"{dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset_name}_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    stats = best_rg_result["heuristic_stats"]

    for heuristic_name, values in stats.items():

        used = values["used"]
        improvements = values["improvements"]
        acceptance_rate = (improvements / used if used > 0 else 0)

        print(f"\n{heuristic_name}")

        print("Used:", used)
        print("Improvements:", improvements)
        print("Acceptance rate:", round(acceptance_rate, 4))
        print("Total gain:", round(values["total_gain"], 2))
        print("Best single gain:", round(values["best_single_gain"], 2))

results_df = pd.DataFrame(all_results)
results_df.to_csv("results_summary.csv", index=False)