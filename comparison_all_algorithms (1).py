import numpy as np
import random
import os
import matplotlib.pyplot as plt

import pso_linear as pso_linear
import pso_fixed as pso_fixed
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
    convergence = []

    for gen in range(NUM_GENERATIONS):
        best_in_pop = de_ea.tools.selBest(population, 1)[0]
        if best_overall is None or best_in_pop.fitness.values[0] < best_overall.fitness.values[0]:
            best_overall = de_ea.creator.Individual(best_in_pop)
            best_overall.fitness.values = best_in_pop.fitness.values

        convergence.append(best_overall.fitness.values[0])

        for i in range(len(population)):
            b_idx, c_idx = de_ea.selectedIndices(len(population), i)
            mutant = de_ea.mutation(best_overall, population[b_idx], population[c_idx], 
                                   de_ea.F, de_ea.MIN_GREEN, de_ea.MAX_GREEN)
            trial = de_ea.crossOver(population[i], mutant, de_ea.CR)
            trial[:] = np.clip(trial, de_ea.MIN_GREEN, de_ea.MAX_GREEN)
            
            trial.fitness.values = de_ea.toolbox.evaluate(trial, traffic_stream)
            if trial.fitness.values[0] < population[i].fitness.values[0]:
                population[i] = trial

    return best_overall.fitness.values[0], convergence

def run_de_rand_engine(seed, traffic_stream):
    
    random.seed(seed)
    np.random.seed(seed)
    
    population = de_ea.toolbox.populationCreator(n=POPULATION_SIZE)
    for ind in population:
        ind.fitness.values = de_ea.toolbox.evaluate(ind, traffic_stream)

    best_so_far = float('inf')
    convergence = []

    for gen in range(NUM_GENERATIONS):
        for i in range(len(population)):
            random_idx = random.sample(range(len(population)), 3)
            a_idx, b_idx, c_idx = random_idx
            mutant = de_ea.mutation(population[a_idx], population[b_idx], population[c_idx], 
                                   de_ea.F, de_ea.MIN_GREEN, de_ea.MAX_GREEN)
            trial = de_ea.crossOver(population[i], mutant, de_ea.CR)
            trial[:] = np.clip(trial, de_ea.MIN_GREEN, de_ea.MAX_GREEN)
            
            trial.fitness.values = de_ea.toolbox.evaluate(trial, traffic_stream)
            if trial.fitness.values[0] < population[i].fitness.values[0]:
                population[i] = trial

        best_in_gen = min(population, key=lambda x: x.fitness.values[0])
        best_so_far = min(best_so_far, best_in_gen.fitness.values[0])
        convergence.append(best_so_far)

    return best_so_far, convergence

def run_pso_linear_engine(seed, traffic_stream):
    
    random.seed(seed)
    np.random.seed(seed)
    
    population = pso_linear.toolbox.populationCreator(n=POPULATION_SIZE)
    global_best = None
    convergence = []

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
        
        convergence.append(global_best.fitness.values[0])
        
        for p in population:
            pso_linear.toolbox.update(p, global_best, current_w)

    return global_best.fitness.values[0], convergence

def run_pso_fixed_engine(seed, traffic_stream):
  
    random.seed(seed)
    np.random.seed(seed)
    
    population = pso_fixed.toolbox.populationCreator(n=POPULATION_SIZE)
    global_best = None
    convergence = []

    for gen in range(NUM_GENERATIONS):
        for p in population:
            p.fitness.values = pso_fixed.toolbox.evaluate(p, traffic_stream)
            
            if p.best is None or p.fitness.values[0] < p.best.fitness.values[0]:
                p.best = pso_fixed.creator.Particle(p)
                p.best.fitness.values = p.fitness.values
            
            if global_best is None or p.fitness.values[0] < global_best.fitness.values[0]:
                global_best = pso_fixed.creator.Particle(p)
                global_best.fitness.values = p.fitness.values
        
        convergence.append(global_best.fitness.values[0])
        
        for p in population:
            pso_fixed.toolbox.update(p, global_best, pso_fixed.W_FIXED)

    return global_best.fitness.values[0], convergence

def generate_improvement_barchart(algorithm_names, improvement_means, improvement_stds, 
                                 output_path="improvement_barchart_comparison.png"):
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#F4A460', '#FF8C00', '#87CEEB', '#4682B4']
    x_pos = np.arange(len(algorithm_names))
    
    bars = ax.bar(x_pos, improvement_means, color=colors, edgecolor='black', 
                  linewidth=1.5, width=0.6, alpha=0.85)
    
    
    for bar, value in zip(bars, improvement_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement %', fontsize=12, fontweight='bold')
    ax.set_title('Average Improvement over Baseline (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithm_names, fontsize=11)
    ax.set_ylim(0, max(improvement_means) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Bar chart saved to: {output_path}")
    plt.close()

def generate_convergence_plot(all_convergence_data, baseline_obj, 
                              output_path="convergence_comparison.png"):
    """Generate convergence plot showing objective value vs generations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#F4A460', '#FF8C00', '#87CEEB', '#4682B4']
    algorithm_names = ['PSO (Fixed W)', 'PSO (Linear W)', 'DE (Rand/1)', 'DE-EA (Best/1)']
    
    generations = np.arange(NUM_GENERATIONS)
    
    for algo_idx, (algo_name, color) in enumerate(zip(algorithm_names, colors)):
        
        avg_convergence = np.mean(all_convergence_data[algo_idx], axis=0)
        ax.plot(generations, avg_convergence, color=color, linewidth=2.5, 
                label=algo_name, marker='o', markersize=4, alpha=0.8)
    
    
    ax.axhline(y=baseline_obj, color='red', linestyle='--', linewidth=2, 
               label='Baseline (60s fixed)', alpha=0.7)
    
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Objective Value', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Comparison: Objective vs Generations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Convergence plot saved to: {output_path}")
    plt.close()

def compare_all_algorithms():
    """Compare all 4 algorithms and generate both bar chart and convergence plot"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ALGORITHM COMPARISON")
    print(f"PSO (Fixed W) vs PSO (Linear W) vs DE (Rand/1) vs DE-EA (Best/1)")
    print(f"{'='*80}\n")

    seeds = [random.randint(1, 99999) for _ in range(NUM_RUNS)]
    
    pso_fixed_results = []
    pso_linear_results = []
    de_rand_results = []
    de_ea_results = []
    
    
    pso_fixed_convergence = []
    pso_linear_convergence = []
    de_rand_convergence = []
    de_ea_convergence = []
    
    baseline_objectives = []

    for i, seed in enumerate(seeds):
        traffic = de_ea.generate_traffic_stream(SIM_HORIZON)
        baseline_obj = de_ea.simulate_traffic(np.full(NUM_INTERSECTIONS * 2, 60.0), traffic)["objective"]
        baseline_objectives.append(baseline_obj)

        
        pso_fixed_val, pso_fixed_conv = run_pso_fixed_engine(seed, traffic)
        pso_linear_val, pso_linear_conv = run_pso_linear_engine(seed, traffic)
        de_rand_val, de_rand_conv = run_de_rand_engine(seed, traffic)
        de_ea_val, de_ea_conv = run_de_ea_engine(seed, traffic)

        
        pso_fixed_convergence.append(pso_fixed_conv)
        pso_linear_convergence.append(pso_linear_conv)
        de_rand_convergence.append(de_rand_conv)
        de_ea_convergence.append(de_ea_conv)

        
        pso_fixed_imp = ((baseline_obj - pso_fixed_val) / baseline_obj) * 100
        pso_linear_imp = ((baseline_obj - pso_linear_val) / baseline_obj) * 100
        de_rand_imp = ((baseline_obj - de_rand_val) / baseline_obj) * 100
        de_ea_imp = ((baseline_obj - de_ea_val) / baseline_obj) * 100

        pso_fixed_results.append(pso_fixed_imp)
        pso_linear_results.append(pso_linear_imp)
        de_rand_results.append(de_rand_imp)
        de_ea_results.append(de_ea_imp)

        print(f"Run {i+1:02d} | PSO-Fixed: {pso_fixed_imp:>6.2f}% | PSO-Linear: {pso_linear_imp:>6.2f}% | "
              f"DE-Rand: {de_rand_imp:>6.2f}% | DE-EA: {de_ea_imp:>6.2f}%")

    
    print(f"\n{'='*80} STATISTICS {'='*80}")
    print(f"PSO (Fixed W) Improvement    : Mean {np.mean(pso_fixed_results):>7.2f}% | Std {np.std(pso_fixed_results):>6.2f}%")
    print(f"PSO (Linear W) Improvement   : Mean {np.mean(pso_linear_results):>7.2f}% | Std {np.std(pso_linear_results):>6.2f}%")
    print(f"DE (Rand/1) Improvement      : Mean {np.mean(de_rand_results):>7.2f}% | Std {np.std(de_rand_results):>6.2f}%")
    print(f"DE-EA (Best/1) Improvement   : Mean {np.mean(de_ea_results):>7.2f}% | Std {np.std(de_ea_results):>6.2f}%")
    
    
    algorithm_names = ['PSO (Fixed W)', 'PSO (Linear W)', 'DE (Rand/1)', 'DE-EA (Best/1)']
    improvement_means = [np.mean(pso_fixed_results), np.mean(pso_linear_results), 
                        np.mean(de_rand_results), np.mean(de_ea_results)]
    improvement_stds = [np.std(pso_fixed_results), np.std(pso_linear_results), 
                       np.std(de_rand_results), np.std(de_ea_results)]
    
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    generate_improvement_barchart(algorithm_names, improvement_means, improvement_stds)
    
    
    avg_baseline = np.mean(baseline_objectives)
    all_convergence_data = [pso_fixed_convergence, pso_linear_convergence, 
                            de_rand_convergence, de_ea_convergence]
    generate_convergence_plot(all_convergence_data, avg_baseline)
    
    
    winner_idx = np.argmax(improvement_means)
    winner = algorithm_names[winner_idx]
    print(f"\n{'='*80}")
    print(f"🏆 OVERALL WINNER: {winner}")
    print(f"   Average Improvement: {improvement_means[winner_idx]:.2f}%")
    print(f"   Standard Deviation: {improvement_stds[winner_idx]:.2f}%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    compare_all_algorithms()
