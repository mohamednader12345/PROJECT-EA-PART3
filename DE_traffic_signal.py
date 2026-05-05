import csv
import json
import os
import random

import numpy as np
from deap import base, creator, tools


from visualizer import plot_results as shared_plot_results
from visualizer import plot_per_run_avg as shared_plot_per_run_avg
from visualizer import plot_individual_runs as shared_plot_individual_runs

F = 0.5
CR = 0.9
NUM_INTERSECTIONS = 2
MIN_GREEN = 10
MAX_GREEN = 120
SERVICE_RATE = 2
CONGESTION_WEIGHT = 120

POPULATION_SIZE = 30
SIM_HORIZON = 150
NUM_GENERATIONS = 50
NUM_RUN = 30
OUTPUT_DIR = os.path.join("de_outputs", "experiment_1") 

toolbox = base.Toolbox()
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

def createIndividual():
    return creator.Individual(np.random.uniform(MIN_GREEN, MAX_GREEN, NUM_INTERSECTIONS * 2))

toolbox.register("individualCreator", createIndividual)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("max", np.max)

def simulate_traffic(time_config, traffic_stream):
    if traffic_stream is None or len(traffic_stream) == 0:
        return {"total_wait": float("inf"), "avg_queue": float("inf"), "objective": float("inf")}

    queue_ns = [0] * NUM_INTERSECTIONS
    queue_ew = [0] * NUM_INTERSECTIONS
    total_wait = 0.0
    queue_accumulator = 0.0

    for current_time in range(len(traffic_stream)):
        arrivals_per_intersection = traffic_stream[current_time]
        for i in range(NUM_INTERSECTIONS):
            green_ns = time_config[2 * i]
            green_ew = time_config[2 * i + 1]
            if green_ns <= 0 or green_ew <= 0:
                return {"total_wait": float("inf"), "avg_queue": float("inf"), "objective": float("inf")}

            arrivals_ns, arrivals_ew = arrivals_per_intersection[i]
            queue_ns[i] += arrivals_ns
            queue_ew[i] += arrivals_ew

            total_cycle = green_ns + green_ew
            time_in_cycle = current_time % int(total_cycle)

            if time_in_cycle < green_ns:
                queue_ns[i] = max(0, queue_ns[i] - SERVICE_RATE)
            else:
                queue_ew[i] = max(0, queue_ew[i] - SERVICE_RATE)

            current_total_queue = queue_ns[i] + queue_ew[i]
            total_wait += current_total_queue
            queue_accumulator += current_total_queue

    avg_queue = queue_accumulator / (len(traffic_stream) * NUM_INTERSECTIONS)
    objective = total_wait + CONGESTION_WEIGHT * avg_queue
    return {"total_wait": float(total_wait), "avg_queue": float(avg_queue), "objective": float(objective)}

def evaluate(individual, traffic_stream):
    results = simulate_traffic(individual, traffic_stream)
    return (results["objective"],)

toolbox.register("evaluate", evaluate)

def generate_traffic_stream(sim_time):
    traffic_stream = []
    for _ in range(sim_time):
        arrivals_snapshot = []
        for _ in range(NUM_INTERSECTIONS):
            base_arrival = random.randint(0, 3)
            arrivals_snapshot.append((base_arrival, base_arrival))
        traffic_stream.append(arrivals_snapshot)
    return traffic_stream

def load_or_create_seeds(num_runs=20):
    random.seed(1) 
    return [random.randint(1, 10000) for _ in range(num_runs)]

def mutation(a, b, c, scale_factor, low, up):
    mutant = c + scale_factor * (b - a)
    mutant = np.clip(mutant, low, up)
    return creator.Individual(mutant)

def crossOver(target, mutant, crossover_rate):
    mask = np.random.rand(len(target)) < crossover_rate
    j_rand = np.random.randint(0, len(target))
    mask[j_rand] = True
    return creator.Individual(np.where(mask, mutant, target))

def selectedIndices(pop_size, current_idx):
    indices = list(range(pop_size))
    indices.remove(current_idx)
    return random.sample(indices, 3)

def save_run_summaries(run_histories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "de_run_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_index", "seed", "baseline_objective", "final_best", "improvement_percent", "best_timings"])
        for run in run_histories:
            baseline = float(run["baseline_objective"])
            final_best = float(run["final_best"])
            improvement = ((baseline - final_best) / baseline) * 100.0 if baseline != 0 else 0.0
            best_timings = "[" + ", ".join(f"{v:.2f}" for v in run["best_solution"]) + "]"
            writer.writerow([run["run_index"], run["seed"], f"{baseline:.6f}", f"{final_best:.6f}", f"{improvement:.4f}", best_timings])
    return csv_path

if __name__ == "__main__":
    seeds = load_or_create_seeds(num_runs=NUM_RUN)
    run_histories = []

    for run_idx, seed_val in enumerate(seeds, start=1):
        random.seed(seed_val)
        np.random.seed(seed_val)

        population = toolbox.populationCreator(n=POPULATION_SIZE)
        traffic_stream = generate_traffic_stream(SIM_HORIZON)

        logbook = tools.Logbook()
        logbook.header = ["gen", "min", "avg", "std", "max"]

        print(f"\-- Run {run_idx} (seed {seed_val}) | Differential Evolution --")

        for individual in population:
            individual.fitness.values = toolbox.evaluate(individual, traffic_stream)

        best_overall = None
        best_curve, avg_curve, wait_curve, queue_curve = [], [], [], []
        best_so_far = float("inf")

        for gen in range(NUM_GENERATIONS):
            for i in range(len(population)):
                a_idx, b_idx, c_idx = selectedIndices(len(population), i)
                mutant = mutation(population[a_idx], population[b_idx], population[c_idx], F, MIN_GREEN, MAX_GREEN)
                trial = crossOver(population[i], mutant, CR)
                trial[:] = np.clip(trial, MIN_GREEN, MAX_GREEN)
                trial.fitness.values = toolbox.evaluate(trial, traffic_stream)

                if trial.fitness.values[0] < population[i].fitness.values[0]:
                    population[i] = trial

            record = stats.compile(population)
            
           
            current_gen_metrics = [simulate_traffic(ind, traffic_stream) for ind in population]
            avg_wait = np.mean([m["total_wait"] for m in current_gen_metrics])
            avg_queue = np.mean([m["avg_queue"] for m in current_gen_metrics])

            best_in_gen_idx = np.argmin([ind.fitness.values[0] for ind in population])
            best_in_gen = population[best_in_gen_idx]
            
            if best_overall is None or best_in_gen.fitness.values[0] < best_overall.fitness.values[0]:
                best_overall = creator.Individual(best_in_gen)
                best_overall.fitness.values = best_in_gen.fitness.values

            best_so_far = min(best_so_far, record["min"])
            
            best_curve.append(best_so_far)
            avg_curve.append(record["avg"])
            wait_curve.append(avg_wait)    
            queue_curve.append(avg_queue) 

            logbook.record(gen=gen, **record)
            print(logbook.stream)

        
        final_metrics = simulate_traffic(best_overall, traffic_stream)

        baseline_timings = np.full(NUM_INTERSECTIONS * 2, 60.0)
        baseline_res = simulate_traffic(baseline_timings, traffic_stream)
        baseline_obj = float(baseline_res["objective"])

        run_histories.append({
            "run_index": run_idx,
            "seed": seed_val,
            "best_curve": best_curve,
            "avg_curve": avg_curve,
            "improvement_curve": [((baseline_obj - v) / baseline_obj) * 100.0 for v in best_curve],
            "wait_curve": wait_curve,
            "queue_curve": queue_curve,
            "baseline_objective": baseline_obj,
            "final_best": float(best_overall.fitness.values[0]),
            "best_solution": np.array(best_overall, dtype=float).tolist(),
            "final_wait": final_metrics["total_wait"],
            "final_avg_queue": final_metrics["avg_queue"]
        })

        print(f"  Baseline objective : {baseline_obj:.2f} | DE best : {best_overall.fitness.values[0]:.2f}")
    best_run = min(run_histories, key=lambda x: x["final_best"])
    print("\n" + "="*50)
    print(f"THE BEST SOLUTION FOUND ACROSS ALL {NUM_RUN} RUNS ")
    print(f"Run Index: {best_run['run_index']}")
    print(f"Best Objective: {best_run['final_best']:.4f}")
    print(f"Total Wait: {best_run['final_wait']:.2f}")
    print(f"Avg Queue: {best_run['final_avg_queue']:.4f}")
    print(f"Best Timings: {best_run['best_solution']}")
    print("="*50)
    shared_plot_individual_runs(run_histories, OUTPUT_DIR, NUM_GENERATIONS, "DE")
    p1, p2, p3 = shared_plot_results(run_histories, OUTPUT_DIR, NUM_GENERATIONS, "DE")
    shared_plot_per_run_avg(run_histories, OUTPUT_DIR, NUM_GENERATIONS, "DE")
    csv_path = save_run_summaries(run_histories, OUTPUT_DIR)


    print(f"\n{'='*60}\nEXPERIMENT COMPLETE\nPlots saved in: {OUTPUT_DIR}\nCSV: {csv_path}\n{'='*60}")