import numpy as np
import random
import os


import DE_traffic_signal as de
import pso_fixed as pso


NUM_GENERATIONS = 50
POPULATION_SIZE = 30
NUM_RUNS = 30
SIM_HORIZON = 150
NUM_INTERSECTIONS = 2

def run_de_engine(seed, traffic_stream):
    
    random.seed(seed)
    np.random.seed(seed)
    
   
    population = de.toolbox.populationCreator(n=POPULATION_SIZE)
    
    for ind in population:
        ind.fitness.values = de.toolbox.evaluate(ind, traffic_stream)

    for gen in range(NUM_GENERATIONS):
        for i in range(len(population)):
            
            a_idx, b_idx, c_idx = de.selectedIndices(len(population), i)
            mutant = de.mutation(population[a_idx], population[b_idx], population[c_idx], de.F, de.MIN_GREEN, de.MAX_GREEN)
            trial = de.crossOver(population[i], mutant, de.CR)
            trial[:] = np.clip(trial, de.MIN_GREEN, de.MAX_GREEN)
            
            trial.fitness.values = de.toolbox.evaluate(trial, traffic_stream)
            if trial.fitness.values[0] < population[i].fitness.values[0]:
                population[i] = trial

    best_ind = min(population, key=lambda x: x.fitness.values[0])
    return best_ind.fitness.values[0]

def run_pso_engine(seed, traffic_stream):
    ""
    random.seed(seed)
    np.random.seed(seed)
    
   
    population = pso.toolbox.populationCreator(n=POPULATION_SIZE)
    global_best = None

    for gen in range(NUM_GENERATIONS):
        for p in population:
            p.fitness.values = pso.toolbox.evaluate(p, traffic_stream)
            
          
            if p.best is None or p.fitness.values[0] < p.best.fitness.values[0]:
                p.best = pso.creator.Particle(p)
                p.best.fitness.values = p.fitness.values
            
            if global_best is None or p.fitness.values[0] < global_best.fitness.values[0]:
                global_best = pso.creator.Particle(p)
                global_best.fitness.values = p.fitness.values
        
        for p in population:
            
            pso.toolbox.update(p, global_best, pso.W_FIXED)

    return global_best.fitness.values[0]

def run_comparison():
    print(f"{'='*50}")
    print(f"STARTING COMPARISON: DE vs PSO")
    print(f"Generations: {NUM_GENERATIONS} | Pop Size: {POPULATION_SIZE} | Runs: {NUM_RUNS}")
    print(f"{'='*50}")

    seeds = [random.randint(1, 10000) for _ in range(NUM_RUNS)]
    
    results = []

    for i, seed in enumerate(seeds):
       
        traffic_stream = de.generate_traffic_stream(SIM_HORIZON)
        
        
        baseline_res = de.simulate_traffic(np.full(NUM_INTERSECTIONS * 2, 60.0), traffic_stream)
        baseline_obj = baseline_res["objective"]

    
        de_best_val = run_de_engine(seed, traffic_stream)
        pso_best_val = run_pso_engine(seed, traffic_stream)

       
        de_impr = ((baseline_obj - de_best_val) / baseline_obj) * 100
        pso_impr = ((baseline_obj - pso_best_val) / baseline_obj) * 100

        results.append({
            'run': i + 1,
            'de_impr': de_impr,
            'pso_impr': pso_impr
        })

        print(f"Run {i+1:02d}: DE Improvement = {de_impr:.2f}% | PSO Improvement = {pso_impr:.2f}%")

    
    avg_de = np.mean([r['de_impr'] for r in results])
    avg_pso = np.mean([r['pso_impr'] for r in results])
    std_de = np.std([r['de_impr'] for r in results])
    std_pso = np.std([r['pso_impr'] for r in results])

    print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    print(f"Average DE Improvement : {avg_de:.2f}% (±{std_de:.2f})")
    print(f"Average PSO Improvement: {avg_pso:.2f}% (±{std_pso:.2f})")
    
    winner = "DE" if avg_de > avg_pso else "PSO"
    print(f"\n🏆 WINNER BASED ON AVG IMPROVEMENT: {winner}")
    print(f"{'='*55}")

if __name__ == "__main__":
    run_comparison()