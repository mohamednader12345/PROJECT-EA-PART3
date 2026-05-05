import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(run_histories, output_dir, num_generations, experiment_label, algo_name="PSO"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    generations = np.arange(num_generations)
    best_curves = np.array([r["best_curve"] for r in run_histories], dtype=float)
    avg_curves = np.array([r["avg_curve"] for r in run_histories], dtype=float)
    imp_curves = np.array([r["improvement_curve"] for r in run_histories], dtype=float)

    mean_best = best_curves.mean(axis=0)
    std_best = best_curves.std(axis=0)
    mean_avg = avg_curves.mean(axis=0)
    mean_imp = imp_curves.mean(axis=0)
    std_imp = imp_curves.std(axis=0)

    baseline_mean = np.mean([r["baseline_objective"] for r in run_histories])
    final_bests = np.array([r["final_best"] for r in run_histories])

    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].plot(generations, mean_best, color="#1f77b4", linewidth=2.5, label="Mean best-so-far")
    axes[0].fill_between(generations, mean_best - std_best, mean_best + std_best, color="#1f77b4", alpha=0.2)
    axes[0].plot(generations, mean_avg, color="#2ca02c", linewidth=2.0, alpha=0.9, label="Mean population avg")
    axes[0].axhline(baseline_mean, linestyle="--", color="#d62728", linewidth=2.0, label="Mean baseline")
   
    axes[0].set_title(f"{algo_name} ({experiment_label}) – Objective Over Generations")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Objective (lower is better)")
    axes[0].legend()

    axes[1].plot(generations, mean_imp, color="#ff7f0e", linewidth=2.5, label="Mean improvement vs baseline")
    axes[1].fill_between(generations, mean_imp - std_imp, mean_imp + std_imp, color="#ff7f0e", alpha=0.2)
    axes[1].axhline(0.0, linestyle="--", color="black", linewidth=1.2)
    axes[1].set_title(f"Improvement Over Baseline ({experiment_label})")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].legend()

    fig.tight_layout()
    p1 = os.path.join(output_dir, "progress_vs_baseline.png")
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)

   
    run_indices = np.arange(1, len(run_histories) + 1)
    baseline_vals = np.array([r["baseline_objective"] for r in run_histories])
    width = 0.42

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.bar(run_indices - width / 2, baseline_vals, width=width, label="Baseline", color="#d62728", alpha=0.85)
    
    ax2.bar(run_indices + width / 2, final_bests, width=width, label=f"{algo_name} final best", color="#1f77b4", alpha=0.90)
    ax2.set_title(f"Per-Run Baseline vs {algo_name} Final Best ({experiment_label})")
    ax2.set_xlabel("Run index")
    ax2.set_ylabel("Objective (lower is better)")
    ax2.legend()
    fig2.tight_layout()
    p2 = os.path.join(output_dir, "baseline_vs_pso_per_run.png")
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)

   
    wait_curves = np.array([r["wait_curve"] for r in run_histories], dtype=float)
    queue_curves = np.array([r["queue_curve"] for r in run_histories], dtype=float)

    mean_wait = wait_curves.mean(axis=0)
    std_wait = wait_curves.std(axis=0)
    mean_queue = queue_curves.mean(axis=0)
    std_queue = queue_curves.std(axis=0)

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))

    axes3[0].plot(generations, mean_wait, color="#1f77b4", linewidth=2.5, label="Mean total wait")
    axes3[0].fill_between(generations, mean_wait - std_wait, mean_wait + std_wait, color="#1f77b4", alpha=0.2)
    axes3[0].set_title(f"Total Waiting Time Over Generations ({experiment_label})")
    axes3[0].set_xlabel("Generation")
    axes3[0].set_ylabel("Total wait (vehicle-seconds)")
    axes3[0].legend()

    axes3[1].plot(generations, mean_queue, color="#2ca02c", linewidth=2.5, label="Mean avg queue")
    axes3[1].fill_between(generations, mean_queue - std_queue, mean_queue + std_queue, color="#2ca02c", alpha=0.2)
    axes3[1].set_title(f"Average Queue Length Over Generations ({experiment_label})")
    axes3[1].set_xlabel("Generation")
    axes3[1].set_ylabel("Avg queue length (vehicles)")
    axes3[1].legend()

    fig3.tight_layout()
    p3 = os.path.join(output_dir, "metrics_over_generations.png")
    fig3.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig3)

    return p1, p2, p3

def plot_individual_runs(run_histories, output_dir, num_generations, experiment_label):
    """تنشئ رسومات منفصلة لكل Run (Progress و Metrics) داخل مجلد خاص."""
    runs_base_dir = os.path.join(output_dir, "individual_runs")
    os.makedirs(runs_base_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="talk")
    generations = np.arange(num_generations)

    for run in run_histories:
        idx = run['run_index']
        run_folder = os.path.join(runs_base_dir, f"run_{idx:02d}")
        os.makedirs(run_folder, exist_ok=True)

        fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7))
        
        axes1[0].plot(generations, run["best_curve"], color="#1f77b4", linewidth=2.5, label="Best-so-far")
        axes1[0].plot(generations, run["avg_curve"], color="#2ca02c", linewidth=2.0, alpha=0.8, label="Population avg")
        axes1[0].axhline(run["baseline_objective"], linestyle="--", color="#d62728", label="Baseline")
        axes1[0].set_title(f"Run {idx} - Objective Progress")
        axes1[0].set_xlabel("Generation")
        axes1[0].set_ylabel("Objective")
        axes1[0].legend()

        axes1[1].plot(generations, run["improvement_curve"], color="#ff7f0e", linewidth=2.5, label="Improvement %")
        axes1[1].axhline(0.0, linestyle="--", color="black", linewidth=1.2)
        axes1[1].set_title(f"Run {idx} - Improvement over Baseline")
        axes1[1].set_xlabel("Generation")
        axes1[1].set_ylabel("Improvement (%)")
        axes1[1].legend()

        fig1.tight_layout()
        fig1.savefig(os.path.join(run_folder, f"run{idx}_progress.png"), dpi=150)
        plt.close(fig1)

        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

        axes2[0].plot(generations, run["wait_curve"], color="#1f77b4", linewidth=2.5)
        axes2[0].set_title(f"Run {idx} - Total Waiting Time")
        axes2[0].set_xlabel("Generation")
        axes2[0].set_ylabel("Total wait (vehicle-seconds)")

        axes2[1].plot(generations, run["queue_curve"], color="#2ca02c", linewidth=2.5)
        axes2[1].set_title(f"Run {idx} - Avg Queue Length")
        axes2[1].set_xlabel("Generation")
        axes2[1].set_ylabel("Avg queue (vehicles)")

        fig2.tight_layout()
        fig2.savefig(os.path.join(run_folder, f"run{idx}_metrics.png"), dpi=150)
        plt.close(fig2)

def plot_per_run_avg(run_histories, output_dir, num_generations, experiment_label, curve_color="#1f77b4"):
    plots_dir = os.path.join(output_dir, "plots") 
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    generations = np.arange(num_generations)

    for run in run_histories:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(generations, run["avg_curve"], color=curve_color, linewidth=2.0)
        ax.set_title(f"Run {run['run_index']} – Population Avg Objective ({experiment_label})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Avg objective")
        fig.tight_layout()
        path = os.path.join(plots_dir, f"run_{run['run_index']:02d}_avg.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)