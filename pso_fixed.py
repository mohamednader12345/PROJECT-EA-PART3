import csv
import json
import os
import random

import numpy as np
from deap import base, creator, tools

from visualizer import plot_results as shared_plot_results
from visualizer import plot_per_run_avg as shared_plot_per_run_avg
from visualizer import plot_individual_runs as shared_plot_individual_runs

W_FIXED = 0.7        

C1 = 2.0
C2 = 2.0
MUTATION_RATE = 0.10

NUM_INTERSECTIONS = 2
MIN_GREEN = 10
MAX_GREEN = 120
V_MAX = 10
SERVICE_RATE = 2    
CONGESTION_WEIGHT = 120

POPULATION_SIZE = 30
SIM_HORIZON = 150
NUM_GENERATIONS = 50
NUM_RUNS = 30
OUTPUT_DIR = os.path.join("pso_outputs", "fixed_inertia")

toolbox = base.Toolbox()

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Particle"):
    creator.create(
        "Particle", np.ndarray,
        fitness=creator.FitnessMin, speed=None, best=None,
        smin=-V_MAX, smax=V_MAX,
    )


def createParticle():
    particle = creator.Particle(
        np.random.uniform(MIN_GREEN, MAX_GREEN, NUM_INTERSECTIONS * 2)
    )
    particle.speed = np.random.uniform(-V_MAX, V_MAX, NUM_INTERSECTIONS * 2)
    return particle


toolbox.register("particleCreator", createParticle)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.particleCreator)

stats = tools.Statistics(lambda p: p.fitness.values[0])
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("max", np.max)


def updateParticle(particle, global_best, inertia_weight):
    r1 = np.random.random(particle.size)
    r2 = np.random.random(particle.size)

    particle.speed = (
        inertia_weight * particle.speed
        + C1 * r1 * (particle.best - particle)
        + C2 * r2 * (global_best - particle)
    )
    particle.speed = np.clip(particle.speed, -V_MAX, V_MAX)
    particle[:] = particle + particle.speed

    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(particle) - 1)
        particle[idx] = random.uniform(MIN_GREEN, MAX_GREEN)

   
    particle[:] = np.clip(particle, MIN_GREEN, MAX_GREEN)



toolbox.register("update", updateParticle)


def simulate_traffic(time, traffic_stream):
    if not traffic_stream:
        return {"total_wait": float("inf"), "avg_queue": float("inf"), "objective": float("inf")}

    queue_ns = [0] * NUM_INTERSECTIONS
    queue_ew = [0] * NUM_INTERSECTIONS
    total_wait = 0.0
    queue_accumulator = 0.0

    for current_time, arrivals_per_intersection in enumerate(traffic_stream):
        for i in range(NUM_INTERSECTIONS):
            green_ns = time[2 * i]
            green_ew = time[2 * i + 1]

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

    return {
        "total_wait": float(total_wait),
        "avg_queue": float(avg_queue),
        "objective": float(objective),
    }


def evaluate(particle, traffic_stream):
    return (simulate_traffic(particle, traffic_stream)["objective"],)


toolbox.register("evaluate", evaluate)


def generate_traffic_stream(sim_time):
    traffic_stream = []
    for _ in range(sim_time):
        snapshot = []
        for _ in range(NUM_INTERSECTIONS):
         
            base_arrival = random.randint(0, 3)
            snapshot.append((base_arrival, base_arrival))
        traffic_stream.append(snapshot)
    return traffic_stream


def load_or_create_seeds( num_runs=30):
    random.seed(1)
    seeds = [random.randint(1, 10000) for _ in range(num_runs)]
    return seeds


def run_single_pso(seed_val, run_idx):
    random.seed(seed_val)
    np.random.seed(seed_val)

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    traffic_stream = generate_traffic_stream(SIM_HORIZON)

    logbook = tools.Logbook()
    logbook.header = ["gen", "w", "min", "avg", "std", "max"]

    best = None
    best_so_far = float("inf")
    best_curve = []
    avg_curve = []
    wait_curve = []
    queue_curve = []

    print(f"\-- Run {run_idx} (seed {seed_val}) | fixed_inertia | W={W_FIXED} --")

    for generation in range(NUM_GENERATIONS):
        for particle in population:
            particle.fitness.values = toolbox.evaluate(particle, traffic_stream)
            if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values
            if best is None or best.size == 0 or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values

        record = stats.compile(population)
        
      
        current_gen_metrics = [simulate_traffic(p, traffic_stream) for p in population]
        avg_wait = np.mean([m["total_wait"] for m in current_gen_metrics])
        avg_queue = np.mean([m["avg_queue"] for m in current_gen_metrics])

        best_so_far = min(best_so_far, record["min"])
        best_curve.append(best_so_far)
        avg_curve.append(record["avg"])
        wait_curve.append(avg_wait)    
        queue_curve.append(avg_queue) 

        logbook.record(gen=generation, w=f"{W_FIXED:.3f}", **record)
        print(logbook.stream)

        for particle in population:
            toolbox.update(particle, best, W_FIXED)

    
    final_best_metrics = simulate_traffic(best, traffic_stream)

    baseline_timings = np.full(NUM_INTERSECTIONS * 2, 60.0)
    baseline_metrics = simulate_traffic(baseline_timings, traffic_stream)
    baseline_objective = float(baseline_metrics["objective"])

    improvement_curve = [
        ((baseline_objective - v) / baseline_objective) * 100.0 if baseline_objective else 0.0
        for v in best_curve
    ]

    print(
        f"  Baseline objective : {baseline_objective:.2f} | "
        f"PSO best : {float(best.fitness.values[0]):.2f}"
    )

    return {
        "run_index": run_idx,
        "seed": seed_val,
        "best_curve": best_curve,
        "avg_curve": avg_curve,
        "improvement_curve": improvement_curve,
        "wait_curve": wait_curve,
        "queue_curve": queue_curve,
        "baseline_objective": baseline_objective,
        "final_best": float(best.fitness.values[0]),
        "final_wait": float(final_best_metrics["total_wait"]),
        "final_avg_queue": float(final_best_metrics["avg_queue"]),
        "best_solution": np.array(best, dtype=float).tolist(),
    }


def plot_results(run_histories):
    shared_plot_individual_runs(run_histories, OUTPUT_DIR, NUM_GENERATIONS, f"Fixed W={W_FIXED}")
    return shared_plot_results(run_histories, OUTPUT_DIR, NUM_GENERATIONS, f"Fixed W={W_FIXED}", algo_name="PSO")


def plot_per_run_avg(run_histories):
    return shared_plot_per_run_avg(
        run_histories,
        OUTPUT_DIR,
        NUM_GENERATIONS,
        f"Fixed W={W_FIXED}",
        curve_color="#ff7f0e",
    )


def save_summaries(run_histories):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "run_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_index", "seed", "algorithm", "config_id", "inertia_scheme",
            "baseline_objective", "final_best", "improvement_percent",
            "final_wait", "final_avg_queue", "best_timings",
        ])
        for run in run_histories:
            baseline   = float(run["baseline_objective"])
            final_best = float(run["final_best"])
            improvement = ((baseline - final_best) / baseline) * 100.0 if baseline else 0.0
            timings = "[" + ", ".join(f"{v:.2f}" for v in run["best_solution"]) + "]"
            writer.writerow([
                run["run_index"], run["seed"], "PSO", "fixed_inertia",
                f"fixed_W={W_FIXED}",
                f"{baseline:.6f}", f"{final_best:.6f}",
                f"{improvement:.4f}",
                f"{run['final_wait']:.2f}", f"{run['final_avg_queue']:.4f}",
                timings,
            ])
    return csv_path


if __name__ == "__main__":
    seeds = load_or_create_seeds( num_runs=NUM_RUNS)

    run_histories = []
    for run_idx, seed_val in enumerate(seeds, start=1):
        result = run_single_pso(seed_val, run_idx)
        run_histories.append(result)

    p1, p2, p3 = plot_results(run_histories)
    csv_path   = save_summaries(run_histories)
    plot_per_run_avg(run_histories)

    final_bests = [r["final_best"] for r in run_histories]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: fixed_inertia  (W={W_FIXED})")
    print(f"  Mean final best : {np.mean(final_bests):.2f}  "
          f"(std {np.std(final_bests):.2f}, min {np.min(final_bests):.2f})")
    print(f"  Plot  : {p1}")
    print(f"  Plot  : {p2}")
    print(f"  Plot  : {p3}")
    print(f"  CSV   : {csv_path}")
    print(f"{'='*60}")