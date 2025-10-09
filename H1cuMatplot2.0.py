import numpy as np
import math
import time
import os


import matplotlib

matplotlib.use('Agg')
# --------------------------------------------------------------------
import matplotlib.pyplot as plt


# ==============================================================================
# 1. FUNCTIILE DE TEST
# ==============================================================================

def rastrigin(x: np.ndarray) -> float:
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * math.pi * x))


def griewangk(x: np.ndarray) -> float:
    sum_part = np.sum(x ** 2 / 4000.0)
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1) + 1e-9)))
    return sum_part - prod_part + 1


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def michalewicz(x: np.ndarray, m: int = 10) -> float:
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x ** 2 / math.pi)) ** (2 * m))


# ==============================================================================
# 2. ALGORITMUL GENETIC HIBRID (SI COMPONENTELE SALE)
# ==============================================================================

def calculate_n_bits(bounds, precision=5):
    """
    NOU: Calculeaza dinamic numarul de biti necesari pentru o precizie data.
    """
    lower_bound, upper_bound = bounds
    # Numarul de "pasi" discreti necesari pentru a acoperi intervalul
    # cu precizia specificata.
    num_steps = (upper_bound - lower_bound) * (10 ** precision)
    # Gasim cel mai mic numar de biti 'n' astfel incat 2^n >= num_steps
    n_bits = math.ceil(math.log2(num_steps))
    return int(n_bits)


def decode_chromosome(bounds: list, n_bits: int, n_dim: int, chromosome: np.ndarray) -> np.ndarray:
    decoded_values = []
    lower_bound, upper_bound = bounds
    for i in range(n_dim):
        start, end = i * n_bits, (i + 1) * n_bits
        bit_substring = chromosome[start:end]
        integer_value = int("".join(bit_substring.astype(str)), 2)
        max_integer = 2 ** n_bits - 1
        scaled_value = lower_bound + (integer_value / max_integer) * (upper_bound - lower_bound)
        decoded_values.append(scaled_value)
    return np.array(decoded_values)


def hill_climbing(objective_func, bounds, n_bits, n_dim, chromosome, max_attempts):
    current_best_chromosome = chromosome.copy()
    current_best_score = objective_func(decode_chromosome(bounds, n_bits, n_dim, current_best_chromosome))
    bit_indices = np.random.permutation(len(chromosome))
    for i in range(min(max_attempts, len(bit_indices))):
        bit_to_flip = bit_indices[i]
        neighbor = current_best_chromosome.copy()
        neighbor[bit_to_flip] = 1 - neighbor[bit_to_flip]
        neighbor_score = objective_func(decode_chromosome(bounds, n_bits, n_dim, neighbor))
        if neighbor_score < current_best_score:
            current_best_chromosome = neighbor
            current_best_score = neighbor_score
    return current_best_chromosome


def selection_tournament_correct(pop, scores, k=3):
    scores = np.asarray(scores)
    tournament_ix = np.random.choice(len(pop), size=k, replace=False)
    tournament_scores = scores[tournament_ix]
    winner_in_tournament_ix = np.argmin(tournament_scores)
    winner_ix = tournament_ix[winner_in_tournament_ix]

    return pop[winner_ix]


def crossover_single_point(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() < r_cross:
        pt = np.random.randint(1, len(p1) - 1)
        c1 = np.concatenate((p1[:pt], p2[pt:]))
        c2 = np.concatenate((p2[:pt], p1[pt:]))
    return [c1, c2]


def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if np.random.rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]


def hybrid_genetic_algorithm(objective_func, bounds, n_dim, n_bits, n_iter, n_pop, r_cross, r_mut, hc_attempts,
                             verbose=True):
    chromosome_len = n_bits * n_dim
    pop = [np.random.randint(0, 2, chromosome_len) for _ in range(n_pop)]
    best_chromosome, best_eval = None, float('inf')
    history = []

    for gen in range(n_iter):
        decoded_pop = [decode_chromosome(bounds, n_bits, n_dim, p) for p in pop]
        scores = [objective_func(d) for d in decoded_pop]

        best_current_idx = np.argmin(scores)

        for i in range(n_pop):
            if scores[i] < best_eval:
                best_chromosome, best_eval = pop[i], scores[i]
                if verbose:
                    print(f"> Gen {gen}, Nou optim: f(x) = {best_eval:.8f}")

        history.append(best_eval)

        # --- CORECTIE APLICATA AICI ---
       
        selected = [selection_tournament_correct(pop, scores) for _ in range(n_pop)]

        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            offspring = crossover_single_point(p1, p2, r_cross)
            for c in offspring:
                mutation(c, r_mut)
                children.append(c)

        elite = pop[best_current_idx]
        refined_elite = hill_climbing(objective_func, bounds, n_bits, n_dim, elite, hc_attempts)

        children_scores = [objective_func(decode_chromosome(bounds, n_bits, n_dim, c)) for c in children]
        worst_child_idx = np.argmax(children_scores)
        children[worst_child_idx] = refined_elite

        pop = children

    best_decoded = decode_chromosome(bounds, n_bits, n_dim, best_chromosome)
    return best_decoded, best_eval, history


# ==============================================================================
# 3. FUNCTII PENTRU VIZUALIZARE SI SALVARE
# ==============================================================================

def plot_convergence(history, title, output_dir="results"):
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='blue', linewidth=2)
    plt.title(f"Grafic de Convergență - {title}", fontsize=16)
    plt.xlabel("Generație", fontsize=12)
    plt.ylabel("Cel mai bun scor (Fitness)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    filename = os.path.join(output_dir, f"convergence_{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Graficul de convergență a fost salvat în: {filename}")


def plot_box_scores(all_scores, title, output_dir="results"):
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_scores, patch_artist=True)
    plt.title(f"Distribuția Scorurilor Finale - {title}", fontsize=16)
    plt.ylabel("Scor Final", fontsize=12)
    plt.xticks([1], ['HGA'])
    plt.grid(True, linestyle='--', alpha=0.6)
    filename = os.path.join(output_dir, f"boxplot_{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Graficul box plot a fost salvat în: {filename}")


def plot_solution_space(objective_func, bounds, solution, title, output_dir="results"):
    if len(solution) != 2: return
    known_optima = {"Rastrigin": [0.0, 0.0], "Griewangk": [0.0, 0.0], "Rosenbrock": [1.0, 1.0],
                    "Michalewicz": [2.20, 1.57]}
    optimum = known_optima.get(title.split(' ')[0], None)
    x = np.linspace(bounds[0], bounds[1], 200)
    y = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([objective_func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Valoarea funcției (Fitness)')
    plt.plot(solution[0], solution[1], 'ro', markersize=10, label='Soluția Găsită')
    if optimum:
        plt.plot(optimum[0], optimum[1], 'w*', markersize=15, label='Optim Global')
    plt.title(f"Spațiul Soluției - {title}", fontsize=16)
    plt.xlabel("x1", fontsize=12)
    plt.ylabel("x2", fontsize=12)
    plt.legend()
    filename = os.path.join(output_dir, f"space_{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Harta spațiului soluției a fost salvată în: {filename}")


def save_summary_to_txt(title, final_scores, avg_duration, best_solution, output_dir="results"):
    summary_content = f"Sumar Rezultate pentru: {title}\n"
    summary_content += "=" * 40 + "\n"
    summary_content += f"Număr de rulări: {len(final_scores)}\n"
    summary_content += f"Timp mediu de execuție: {avg_duration:.2f} secunde/rulare\n\n"
    summary_content += "Statistici Scoruri:\n"
    summary_content += f"  - Cel mai bun scor: {np.min(final_scores):.8f}\n"
    summary_content += f"  - Scorul mediu:    {np.mean(final_scores):.8f}\n"
    summary_content += f"  - Deviația standard: {np.std(final_scores):.8f}\n"
    summary_content += f"  - Cel mai slab scor: {np.max(final_scores):.8f}\n\n"
    summary_content += "Cea mai bună soluție găsită (vectorul x):\n"
    solution_str = np.array2string(best_solution, precision=5, separator=', ')
    summary_content += f"  {solution_str}\n"
    summary_content += "=" * 40 + "\n"
    filename = os.path.join(output_dir, f"summary_{title.replace(' ', '_')}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    print(f"Sumarul text a fost salvat în: {filename}")


# ==============================================================================
# 4. BLOCUL DE EXECUTIE PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    # --- Parametrii Algoritmului ---
    PRECISION = 5  # Numarul de zecimale de precizie dorit
    N_ITER = 200
    N_POP = 100
    R_CROSS = 0.9
    HC_ATTEMPTS = 20
    N_RUNS = 30

    if not os.path.exists("results"):
        os.makedirs("results")

    functions_to_test = {
        "Rastrigin": (rastrigin, [-5.12, 5.12]),
        "Griewangk": (griewangk, [-600.0, 600.0]),
        "Rosenbrock": (rosenbrock, [-5.0, 10.0]),
        "Michalewicz": (michalewicz, [0, math.pi])
    }

    dimensions_to_test = [2, 30, 100]

    for func_name, (func, bounds) in functions_to_test.items():
        
        n_bits = calculate_n_bits(bounds, PRECISION)

        for dim in dimensions_to_test:
            title = f"{func_name} D={dim}"
            print("\n" + "=" * 70)
            print(f"Execuție experiment pentru: {title}")
            print(f"Precizie: {PRECISION} zecimale -> Numar de biti calculat: {n_bits}")
            print(f"Se vor efectua {N_RUNS} rulări pentru statistică...")
            print("=" * 70)

            final_scores = []
            best_run_solution = None
            best_run_score = float('inf')
            best_run_history = None

            start_time = time.time()
            for r in range(N_RUNS):
                print(f"  Rularea {r + 1}/{N_RUNS}...")
                R_MUT = 1.0 / (n_bits * dim)
                solution, score, history = hybrid_genetic_algorithm(
                    objective_func=func, bounds=bounds, n_dim=dim, n_bits=n_bits,
                    n_iter=N_ITER, n_pop=N_POP, r_cross=R_CROSS, r_mut=R_MUT,
                    hc_attempts=HC_ATTEMPTS, verbose=False
                )
                final_scores.append(score)
                if score < best_run_score:
                    best_run_score = score
                    best_run_solution = solution
                    best_run_history = history

            avg_duration = (time.time() - start_time) / N_RUNS

            print("\n" + "-" * 70)
            print(f"REZULTATE FINALE pentru {title}:")
            print(f"  -> Timp mediu de execuție: {avg_duration:.2f} secunde/rulare")
            print(f"  -> Cel mai bun scor obținut (din {N_RUNS} rulări): {np.min(final_scores):.8f}")
            print(f"  -> Scorul mediu: {np.mean(final_scores):.8f}")
            print(f"  -> Deviația standard: {np.std(final_scores):.8f}")
            if dim > 5:
                print(f"  -> Solutia (primele 5 componente): {best_run_solution[:5]}")
            else:
                print(f"  -> Solutia (x): {best_run_solution}")
            print("-" * 70)

            # --- Generam vizualizarile si sumarul text ---
            plot_convergence(best_run_history, title)
            plot_box_scores(final_scores, title)
            if dim == 2:
                plot_solution_space(func, bounds, best_run_solution, title)

            save_summary_to_txt(title, final_scores, avg_duration, best_run_solution)
