import numpy as np
import random
import os


import pso_linear as pso_linear
import de2 as de_ea


NUM_GENERATIONS = 50
POPULATION_SIZE = 30
NUM_RUNS = 30 
SIM_HORIZON = 150
NUM_INTERSECTIONS = 2

def run_de_ea_engine(seed, traffic_stream):
    
    random.seed(seed)
    np.random.seed(seed)
    
    
    population = de_ea.toolbox.populationCreator(n=POPULATION_SIZE)
    for ind in population:
        ind.fitness.values = de_ea.toolbox.evaluate(ind, traffic_stream)

    best_overall = None

    for gen in range(NUM_GENERATIONS):
      
        best_in_pop = de_ea.tools.selBest(population, 1)[0]
        if best_overall is None or best_in_pop.fitness.values[0] < best_overall.fitness.values[0]:
            best_overall = de_ea.creator.Individual(best_in_pop)
            best_overall.fitness.values = best_in_pop.fitness.values

        for i in range(len(population)):
           
            b_idx, c_idx = de_ea.selectedIndices(len(population), i)
            mutant = de_ea.mutation(best_overall, population[b_idx], population[c_idx], de_ea.F, de_ea.MIN_GREEN, de_ea.MAX_GREEN)
            trial = de_ea.crossOver(population[i], mutant, de_ea.CR)
            trial[:] = np.clip(trial, de_ea.MIN_GREEN, de_ea.MAX_GREEN)
            
            trial.fitness.values = de_ea.toolbox.evaluate(trial, traffic_stream)
            if trial.fitness.values[0] < population[i].fitness.values[0]:
                population[i] = trial

    return best_overall.fitness.values[0]

def run_pso_linear_engine(seed, traffic_stream):
    
    random.seed(seed)
    np.random.seed(seed)
    
    population = pso_linear.toolbox.populationCreator(n=POPULATION_SIZE)
    global_best = None

    for gen in range(NUM_GENERATIONS):
      
        current_w = pso_linear.get_inertia(gen, NUM_GENERATIONS)
        
        for p in population:
            p.fitness.values = pso_linear.toolbox.evaluate(p, traffic_stream)
            
            
            if p.best is None or p.fitness.values[0] < p.best.fitness.values[0]:
                p.best = pso_linear.creator.Particle(p)
                p.best.fitness.values = p.fitness.values
            
            if global_best is None or p.fitness.values[0] < global_best.fitness.values[0]:
                global_best = pso_linear.creator.Particle(p)
                global_best.fitness.values = p.fitness.values
        
        
        for p in population:
            pso_linear.toolbox.update(p, global_best, current_w)

    return global_best.fitness.values[0]

def compare_algorithms():
    print(f"{'='*60}")
    print(f"COMPARISON: DE-EA (Best-Mutation) vs PSO (Linear Inertia)")
    print(f"{'='*60}")

    
    seeds = [random.randint(1, 99999) for _ in range(NUM_RUNS)]
    
    de_results = []
    pso_results = []

    for i, seed in enumerate(seeds):
        
        traffic = de_ea.generate_traffic_stream(SIM_HORIZON)
        
        
        baseline_obj = de_ea.simulate_traffic(np.full(NUM_INTERSECTIONS * 2, 60.0), traffic)["objective"]

      
        de_val = run_de_ea_engine(seed, traffic)
        pso_val = run_pso_linear_engine(seed, traffic)

      
        de_imp = ((baseline_obj - de_val) / baseline_obj) * 100
        pso_imp = ((baseline_obj - pso_val) / baseline_obj) * 100

        de_results.append(de_imp)
        pso_results.append(pso_imp)

        print(f"Run {i+1:02d} | Seed: {seed:<5} | DE-EA: {de_imp:>6.2f}% | PSO-Linear: {pso_imp:>6.2f}%")

   
    print(f"\n{'='*25} STATISTICS {'='*25}")
    print(f"DE-EA (Updated) Improvement  : Mean {np.mean(de_results):.2f}% | Std {np.std(de_results):.2f}%")
    print(f"PSO (Linear W) Improvement   : Mean {np.mean(pso_results):.2f}% | Std {np.std(pso_results):.2f}%")
    
    winner = "DE-EA" if np.mean(de_results) > np.mean(pso_results) else "PSO-Linear"
    print(f"\n🏆 OVERALL WINNER: {winner}")
    print(f"{'='*62}")

if __name__ == "__main__":
    compare_algorithms()