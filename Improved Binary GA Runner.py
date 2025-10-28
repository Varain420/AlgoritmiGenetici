import numpy as np
import math
import time
import os
import shutil
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# 1. FUNCTIILE DE TEST (neschimbate)
# ==============================================================================
def rastrigin(x: np.ndarray) -> float:
    A = 10
    n_dim = x.shape[1] if x.ndim > 1 else len(x)
    return A * n_dim + np.sum(x ** 2 - A * np.cos(2 * math.pi * x), axis=1 if x.ndim > 1 else None)


def griewangk(x: np.ndarray) -> float:
    n_dim = x.shape[1] if x.ndim > 1 else len(x)
    indices = np.arange(1, n_dim + 1)
    sum_part = np.sum(x ** 2 / 4000.0, axis=1 if x.ndim > 1 else None)
    prod_part = np.prod(np.cos(x / np.sqrt(indices)), axis=1 if x.ndim > 1 else None)
    return sum_part - prod_part + 1


def rosenbrock(x: np.ndarray) -> float:
    if x.ndim > 1:
        return np.array([rosenbrock(ind) for ind in x])
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def michalewicz(x: np.ndarray, m: int = 10) -> float:
    n_dim = x.shape[1] if x.ndim > 1 else len(x)
    i = np.arange(1, n_dim + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x ** 2 / math.pi)) ** (2 * m), axis=1 if x.ndim > 1 else None)


# ==============================================================================
# 2. ALGORITMUL GENETIC HIBRID (VERSIUNE BINARA IMBUNATATITA)
# ==============================================================================

def calculate_n_bits(bounds, precision=5):
    lower_bound, upper_bound = bounds
    num_steps = (upper_bound - lower_bound) * (10 ** precision)
    n_bits = math.ceil(math.log2(num_steps))
    return int(n_bits)


# --- Functii pentru Codificarea Gray ---
def gray_to_binary_vectorized(gray_codes):
    # Converteste o matrice de intregi in cod Gray la intregi binari standard.
    # gray_codes este o matrice de intregi (fiecare rand e un numar)
    mask = gray_codes >> 1
    binary_codes = gray_codes.copy()
    while np.any(mask):
        binary_codes ^= mask
        mask >>= 1
    return binary_codes


# --- Decodificare adaptata pentru Codificarea Gray ---
def decode_population_gray_vectorized(bounds, n_bits, n_dim, population):
    lower_bound, upper_bound = bounds
    pop_reshaped = population.reshape(population.shape[0], n_dim, n_bits)

    # Convertim din binar in intregi (interpretati ca fiind in cod Gray)
    powers_of_2 = 2 ** np.arange(n_bits - 1, -1, -1)
    gray_integer_values = pop_reshaped @ powers_of_2

    # Convertim din intregi Gray in intregi binari standard
    binary_integer_values = gray_to_binary_vectorized(gray_integer_values)

    # Scalam valorile la intervalul dorit
    max_integer = 2 ** n_bits - 1
    scaled_values = lower_bound + (binary_integer_values / max_integer) * (upper_bound - lower_bound)

    return scaled_values


# --- Hill Climbing adaptat pentru a folosi decodificarea Gray ---
def hill_climbing_gray(objective_func, bounds, n_bits, n_dim, chromosome, max_attempts):
    def decode_single_gray(chrom):
        pop_matrix = chrom.reshape(1, -1)
        return decode_population_gray_vectorized(bounds, n_bits, n_dim, pop_matrix)[0]

    current_best_chromosome = chromosome.copy()
    current_best_score = objective_func(decode_single_gray(current_best_chromosome))

    bit_indices = np.random.permutation(len(chromosome))
    for i in range(min(max_attempts, len(bit_indices))):
        bit_to_flip = bit_indices[i]
        neighbor = current_best_chromosome.copy()
        neighbor[bit_to_flip] = 1 - neighbor[bit_to_flip]

        neighbor_score = objective_func(decode_single_gray(neighbor))

        if neighbor_score < current_best_score:
            current_best_chromosome = neighbor
            current_best_score = neighbor_score

    return current_best_chromosome


# --- Operatori si selectie  ---
def crossover_uniform_vectorized(p1, p2, r_cross):
    if np.random.rand() >= r_cross:
        return p1.copy(), p2.copy()
    mask = np.random.rand(len(p1)) < 0.5
    c1 = np.where(mask, p2, p1)
    c2 = np.where(mask, p1, p2)
    return c1, c2


def mutation_vectorized(chromosome, r_mut):
    mask = np.random.rand(len(chromosome)) < r_mut
    chromosome[mask] = 1 - chromosome[mask]
    return chromosome


def selection_tournament(pop, scores, k=3):
    pop_indices = np.arange(len(pop))
    tournament_ix = np.random.choice(pop_indices, size=k, replace=False)
    winner_local_ix = np.argmin(scores[tournament_ix])
    winner_global_ix = tournament_ix[winner_local_ix]
    return pop[winner_global_ix]


def selection_roulette_wheel(pop, scores):
    inverted_scores = 1.0 / (scores - np.min(scores) + 1e-9)
    total_fitness = np.sum(inverted_scores)
    if total_fitness == 0:
        probabilities = np.full(len(pop), 1 / len(pop))
    else:
        probabilities = inverted_scores / total_fitness
    selected_ix = np.random.choice(len(pop), p=probabilities)
    return pop[selected_ix]


# --- Algoritmul principal, cu Mutație Non-Uniformă ---
def bga_improved(objective_func, bounds, n_dim, n_bits, n_iter, n_pop, r_cross, r_mut_initial, hc_attempts,
                 selection_func, stagnation_limit=None):
    chromosome_len = n_bits * n_dim
    pop = np.random.randint(0, 2, size=(n_pop, chromosome_len))

    best_chromosome, best_eval = None, float('inf')
    history = []
    stagnation_counter = 0

    # Parametri pentru mutatia non-uniforma
    r_mut_final = r_mut_initial / 100  # Rata finala, foarte mica
    decay_factor = 4.0  # Cat de repede scade rata de mutatie

    for gen in range(n_iter):
        # 1. Calculam rata de mutatie pentru generatia curenta
        progress = gen / n_iter
        effective_r_mut = r_mut_final + (r_mut_initial - r_mut_final) * (1 - progress) ** decay_factor

        # 2. Decodificare si Evaluare (folosind functia Gray)
        decoded_pop = decode_population_gray_vectorized(bounds, n_bits, n_dim, pop)
        scores = objective_func(decoded_pop)

        best_current_idx = np.argmin(scores)

        if scores[best_current_idx] < best_eval:
            best_chromosome, best_eval = pop[best_current_idx].copy(), scores[best_current_idx]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        history.append(best_eval)

        # 3. Re-initializare la stagnare
        if stagnation_limit is not None and stagnation_counter >= stagnation_limit:
            stagnation_counter = 0
            n_elite = int(0.1 * n_pop)
            sorted_indices = np.argsort(scores)
            elite_pop = pop[sorted_indices[:n_elite]]
            new_individuals = np.random.randint(0, 2, size=(n_pop - n_elite, chromosome_len))
            pop = np.vstack([elite_pop, new_individuals])
            decoded_pop = decode_population_gray_vectorized(bounds, n_bits, n_dim, pop)
            scores = objective_func(decoded_pop)
            best_current_idx = np.argmin(scores)

        # 4. Selectie
        selected_pop = np.array([selection_func(pop, scores) for _ in range(n_pop)])

        # 5. Crossover si Mutatie (cu rata de mutatie dinamica)
        children = np.empty_like(pop)
        for i in range(0, n_pop, 2):
            p1, p2 = selected_pop[i], selected_pop[i + 1]
            c1, c2 = crossover_uniform_vectorized(p1, p2, r_cross)
            children[i] = mutation_vectorized(c1, effective_r_mut)
            children[i + 1] = mutation_vectorized(c2, effective_r_mut)

        # 6. Elitism si Hill Climbing (folosind functia Gray)
        elite = pop[best_current_idx]
        refined_elite = hill_climbing_gray(objective_func, bounds, n_bits, n_dim, elite, hc_attempts)

        decoded_children = decode_population_gray_vectorized(bounds, n_bits, n_dim, children)
        children_scores = objective_func(decoded_children)
        worst_child_idx = np.argmax(children_scores)
        children[worst_child_idx] = refined_elite

        pop = children

    best_decoded = decode_population_gray_vectorized(bounds, n_bits, n_dim, best_chromosome.reshape(1, -1))[0]
    return best_decoded, best_eval, history


# ==============================================================================
# 3. FUNCTII PENTRU VIZUALIZARE SI SALVARE
# ==============================================================================
def plot_convergence(history, title, output_dir):
    plt.figure(figsize=(10, 6));
    plt.plot(history)
    plt.title(f"Convergenta - {title}");
    plt.xlabel("Generatie");
    plt.ylabel("Cel mai bun scor")
    plt.grid(True);
    plt.savefig(os.path.join(output_dir, "convergence.png"));
    plt.close()


def plot_box_scores(all_scores, title, output_dir):
    plt.figure(figsize=(8, 6));
    plt.boxplot(all_scores)
    plt.title(f"Distributia Scorurilor - {title}");
    plt.ylabel("Scor Final")
    plt.xticks([1], ['BGA Îmbunătățit']);
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "boxplot.png"));
    plt.close()


def save_summary_to_txt(title, final_scores, avg_duration, best_solution, params, output_dir):
    summary_content = f"Sumar Rezultate pentru: {title}\n" + "=" * 40 + "\n"
    summary_content += "Parametrii Experiment:\n"
    for key, value in params.items(): summary_content += f"  - {key}: {value}\n"
    summary_content += "=" * 40 + "\n"
    summary_content += f"Numar de rulari: {len(final_scores)}\n"
    summary_content += f"Timp mediu de executie: {avg_duration:.2f} secunde/rulare\n\n"
    summary_content += "Statistici Scoruri:\n"
    summary_content += f"  - Cel mai bun scor: {np.min(final_scores):.8f}\n"
    summary_content += f"  - Scorul mediu:    {np.mean(final_scores):.8f}\n"
    summary_content += f"  - Dev. standard:   {np.std(final_scores):.8f}\n\n"
    summary_content += "Cea mai buna solutie gasita (vectorul x):\n"
    solution_str = np.array2string(best_solution, precision=5, separator=', ')
    summary_content += f"  {solution_str}\n" + "=" * 40 + "\n"
    with open(os.path.join(output_dir, "summary.txt"), 'w', encoding='utf-8') as f: f.write(summary_content)


# ==============================================================================
# 4. BLOCUL DE EXECUTIE PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    PRECISION = 4
    N_ITER_BASE, N_POP_BASE, HC_ATTEMPTS_BASE = 200, 100, 15
    N_ITER_HIGH_DIM, N_POP_HIGH_DIM, HC_ATTEMPTS_HIGH_DIM = 400, 150, 20
    STAGNATION_LIMIT = 50
    N_RUNS = 30
    MAIN_OUTPUT_DIR = "GRID_SEARCH_RESULTS_BINAR_IMBUNATATIT"

    CROSSOVER_RATES = [0.7, 0.8, 0.9]
    MUTATION_MULTIPLIERS = [1.0, 2.0, 4.0]
    SELECTION_METHODS = {"Tournament": selection_tournament, "Roulette": selection_roulette_wheel}

    FUNCTIONS_TO_TEST = {
        "Rastrigin": (rastrigin, [-5.12, 5.12]),
        "Griewangk": (griewangk, [-600.0, 600.0]),
        "Rosenbrock": (rosenbrock, [-5.0, 10.0]),
        "Michalewicz": (michalewicz, [0, math.pi])
    }
    DIMENSIONS_TO_TEST = [2, 30, 100]

    if os.path.exists(MAIN_OUTPUT_DIR): shutil.rmtree(MAIN_OUTPUT_DIR)
    os.makedirs(MAIN_OUTPUT_DIR)

    total_experiments = len(FUNCTIONS_TO_TEST) * len(DIMENSIONS_TO_TEST) * len(SELECTION_METHODS) * len(
        CROSSOVER_RATES) * len(MUTATION_MULTIPLIERS)
    current_experiment = 0

    for func_name, (func, bounds) in FUNCTIONS_TO_TEST.items():
        n_bits = calculate_n_bits(bounds, PRECISION)
        for dim in DIMENSIONS_TO_TEST:
            if dim >= 30:
                n_iter, n_pop, hc_attempts = N_ITER_HIGH_DIM, N_POP_HIGH_DIM, HC_ATTEMPTS_HIGH_DIM
                stagnation_to_use = STAGNATION_LIMIT
            else:
                n_iter, n_pop, hc_attempts = N_ITER_BASE, N_POP_BASE, HC_ATTEMPTS_BASE
                stagnation_to_use = None

            for sel_name, sel_func in SELECTION_METHODS.items():
                for r_cross in CROSSOVER_RATES:
                    for mut_multiplier in MUTATION_MULTIPLIERS:
                        current_experiment += 1
                        r_mut_base = 1.0 / (n_bits * dim)
                        r_mut = r_mut_base * mut_multiplier
                        exp_name = f"{func_name}_D{dim}_{sel_name}_C{r_cross}_M{mut_multiplier:.1f}"
                        output_dir = os.path.join(MAIN_OUTPUT_DIR, exp_name)
                        os.makedirs(output_dir)

                        print(f"\n[{current_experiment}/{total_experiments}] RULEZ EXPERIMENT: {exp_name}")

                        final_scores, best_run_solution, best_run_score, best_run_history = [], None, float('inf'), None
                        start_time = time.time()

                        for r in range(N_RUNS):
                            solution, score, history = bga_improved(
                                objective_func=func, bounds=bounds, n_dim=dim, n_bits=n_bits,
                                n_iter=n_iter, n_pop=n_pop, r_cross=r_cross, r_mut_initial=r_mut,
                                hc_attempts=hc_attempts, selection_func=sel_func,
                                stagnation_limit=stagnation_to_use,
                            )
                            final_scores.append(score)
                            if score < best_run_score:
                                best_run_score, best_run_solution, best_run_history = score, solution, history

                        avg_duration = (time.time() - start_time) / N_RUNS

                        title = f"{func_name} D={dim} ({sel_name})"
                        params = {
                            "Reprezentare": "Binar Îmbunătățit (Gray + Non-Uniform Mut.)",
                            "Functie": func_name, "Dimensiune": dim, "Selectie": sel_name,
                            "r_cross": r_cross, "r_mut_initial": f"{r_mut:.6f} (Multiplier: {mut_multiplier})",
                            "n_iter": n_iter, "n_pop": n_pop,
                            "stagnation_limit": stagnation_to_use if stagnation_to_use else "None"
                        }

                        save_summary_to_txt(title, final_scores, avg_duration, best_run_solution, params, output_dir)
                        plot_convergence(best_run_history, title, output_dir)
                        plot_box_scores(final_scores, title, output_dir)

                        print(f"--> Finalizat. Timp mediu: {avg_duration:.2f}s. Rezultate salvate in: {output_dir}")

    print("\n\nGrid search complet!")

