import numpy as np
import math
import time


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




def calculate_n_bits(bounds, precision=5):

    lower_bound, upper_bound = bounds
    num_steps = (upper_bound - lower_bound) * (10 ** precision)
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


def selection_tournament(pop, scores, k=3):
    selection_ix = np.random.randint(len(pop))
    for _ in range(k - 1):
        ix = np.random.randint(len(pop))
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


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


    for gen in range(n_iter):
        decoded_pop = [decode_chromosome(bounds, n_bits, n_dim, p) for p in pop]
        scores = [objective_func(d) for d in decoded_pop]

        best_current_idx = np.argmin(scores)

        for i in range(n_pop):
            if scores[i] < best_eval:
                best_chromosome, best_eval = pop[i], scores[i]
                if verbose:
                    print(f"> Gen {gen}, Nou optim: f(x) = {best_eval:.8f}")

        selected = [selection_tournament(pop, scores) for _ in range(n_pop)]

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
    return best_decoded, best_eval




if __name__ == '__main__':

    PRECISION = 5
    N_ITER = 200
    N_POP = 100
    R_CROSS = 0.9
    HC_ATTEMPTS = 20
    N_RUNS = 30

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


            start_time = time.time()
            for r in range(N_RUNS):
                print(f"  Rularea {r + 1}/{N_RUNS}...")
                R_MUT = 1.0 / (n_bits * dim)
                solution, score = hybrid_genetic_algorithm(
                    objective_func=func, bounds=bounds, n_dim=dim, n_bits=n_bits,
                    n_iter=N_ITER, n_pop=N_POP, r_cross=R_CROSS, r_mut=R_MUT,
                    hc_attempts=HC_ATTEMPTS, verbose=False
                )
                final_scores.append(score)
                if score < best_run_score:
                    best_run_score = score
                    best_run_solution = solution


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
