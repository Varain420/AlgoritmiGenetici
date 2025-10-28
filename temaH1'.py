import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os

# ==============================================================================
# CONFIGURARE SALVARE
# ==============================================================================
# Creează folder pentru rezultate în același director cu scriptul
OUTPUT_DIR = "hill_climbing_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ==============================================================================
# FUNCTII DE BAZA
# ==============================================================================

def f(x):

    return x ** 3 - 60 * x ** 2 + 900 * x + 100


def int_to_binary(n, bits=5):

    return format(n, f'0{bits}b')


def binary_to_int(binary_str):

    return int(binary_str, 2)


def get_neighbors(x):

    binary = int_to_binary(x)
    neighbors = []

    for i in range(5):
        # Flip bit-ul i
        neighbor_binary = list(binary)
        neighbor_binary[i] = '1' if neighbor_binary[i] == '0' else '0'
        neighbor_binary = ''.join(neighbor_binary)
        neighbor_int = binary_to_int(neighbor_binary)
        neighbors.append(neighbor_int)

    return neighbors


# ==============================================================================
# ALGORITMI HILL CLIMBING
# ==============================================================================

def hill_climbing_first_improvement(start):

    current = start
    path = [current]

    while True:
        neighbors = get_neighbors(current)
        improved = False

        for neighbor in neighbors:
            if f(neighbor) > f(current):
                current = neighbor
                path.append(current)
                improved = True
                break  # Prima îmbunătățire

        if not improved:
            break

    return current, path


def hill_climbing_best_improvement(start):

    current = start
    path = [current]

    while True:
        neighbors = get_neighbors(current)
        best_neighbor = current
        best_fitness = f(current)

        for neighbor in neighbors:
            if f(neighbor) > best_fitness:
                best_neighbor = neighbor
                best_fitness = f(neighbor)

        if best_neighbor == current:
            break

        current = best_neighbor
        path.append(current)

    return current, path


# ==============================================================================
# ANALIZA BAZINELOR
# ==============================================================================

def analyze_basins(method='first'):

    basins = {}
    all_paths = {}

    for start in range(32):
        if method == 'first':
            optimum, path = hill_climbing_first_improvement(start)
        else:
            optimum, path = hill_climbing_best_improvement(start)

        all_paths[start] = path

        if optimum not in basins:
            basins[optimum] = []
        basins[optimum].append(start)

    return basins, all_paths


def find_local_maxima():

    local_maxima = []

    for x in range(32):
        neighbors = get_neighbors(x)
        is_local_max = all(f(x) >= f(neighbor) for neighbor in neighbors)

        if is_local_max:
            local_maxima.append(x)

    return local_maxima


# ==============================================================================
# VIZUALIZARE
# ==============================================================================

def create_fitness_landscape():

    x_values = np.arange(32)
    fitness_values = [f(x) for x in x_values]
    local_maxima = find_local_maxima()

    colors = {7: '#FF6B6B', 10: '#4ECDC4', 12: '#45B7D1', 16: '#FFA07A'}
    labels = {7: 'x=7 (f=3803)', 10: 'x=10 (f=4100) *', 12: 'x=12 (f=3988)', 16: 'x=16 (f=3236)'}

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, fitness_values, 'b-', linewidth=2, alpha=0.7)
    plt.scatter(x_values, fitness_values, c='blue', s=50, alpha=0.5, zorder=5)

    # Marcare maxime locale
    for opt in local_maxima:
        marker = '*' if opt == 10 else 'o'
        size = 300 if opt == 10 else 200
        plt.scatter(opt, f(opt), c=colors[opt], s=size, marker=marker,
                    edgecolors='black', linewidths=2, zorder=10, label=labels[opt])

    plt.xlabel('x (reprezentare zecimală)', fontsize=11, fontweight='bold')
    plt.ylabel('f(x)', fontsize=11, fontweight='bold')
    plt.title('Relieful Fitness și Maximele Locale', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=9)
    plt.xticks(range(0, 32, 2))

    plt.savefig(os.path.join(OUTPUT_DIR, 'fitness_landscape.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvat: {OUTPUT_DIR}/fitness_landscape.png")


def create_basins_comparison():

    basins_first, _ = analyze_basins('first')
    basins_best, _ = analyze_basins('best')

    colors = {7: '#FF6B6B', 10: '#4ECDC4', 12: '#45B7D1', 16: '#FFA07A'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # First Improvement
    for opt in sorted(basins_first.keys()):
        basin = basins_first[opt]
        for x in basin:
            ax1.bar(x, f(x), color=colors[opt], alpha=0.7, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('x (puncte de start)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax1.set_title('Bazine de Atracție - FIRST IMPROVEMENT', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(range(0, 32, 2))

    # Best Improvement
    for opt in sorted(basins_best.keys()):
        basin = basins_best[opt]
        for x in basin:
            ax2.bar(x, f(x), color=colors[opt], alpha=0.7, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('x (puncte de start)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax2.set_title('Bazine de Atracție - BEST IMPROVEMENT', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(0, 32, 2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'basins_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvat: {OUTPUT_DIR}/basins_comparison.png")


def create_transition_graph(basins, paths, method_name):

    plt.figure(figsize=(14, 10))

    # Calculăm pozițiile pentru toate punctele (aranjare circulară)
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    positions = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(32)}

    colors = {7: '#FF6B6B', 10: '#4ECDC4', 12: '#45B7D1', 16: '#FFA07A'}

    # Desenăm tranzițiile
    for start in range(32):
        path = paths[start]
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]

            x_coords = [positions[current][0], positions[next_point][0]]
            y_coords = [positions[current][1], positions[next_point][1]]

            optimum = path[-1]
            plt.arrow(x_coords[0], y_coords[0],
                      x_coords[1] - x_coords[0], y_coords[1] - y_coords[0],
                      head_width=0.03, head_length=0.05, fc=colors[optimum],
                      ec=colors[optimum], alpha=0.3, linewidth=0.5)

    # Desenăm nodurile
    for i in range(32):
        optimum = paths[i][-1]
        is_optimum = (i in basins.keys())

        if is_optimum:
            plt.scatter(positions[i][0], positions[i][1],
                        c=colors[i], s=500, marker='*',
                        edgecolors='black', linewidths=2, zorder=10)
        else:
            plt.scatter(positions[i][0], positions[i][1],
                        c=colors[optimum], s=200, alpha=0.7,
                        edgecolors='black', linewidths=1, zorder=5)

        plt.text(positions[i][0] * 1.15, positions[i][1] * 1.15, str(i),
                 ha='center', va='center', fontsize=8, fontweight='bold')

    plt.title(f'Graf de Tranziții - {method_name}', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.axis('off')

    plt.savefig(os.path.join(OUTPUT_DIR, f'transitions_{method_name.lower().replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvat: {OUTPUT_DIR}/transitions_{method_name.lower().replace(' ', '_')}.png")


def create_convergence_examples():

    basins_first, paths_first = analyze_basins('first')
    basins_best, paths_best = analyze_basins('best')

    example_starts = [0, 6, 8, 15, 20, 22]
    x_values = np.arange(32)
    fitness_values = [f(x) for x in x_values]

    fig = plt.figure(figsize=(16, 10))

    for idx, start in enumerate(example_starts):
        ax = plt.subplot(2, 3, idx + 1)

        # Desenăm funcția
        ax.plot(x_values, fitness_values, 'gray', linewidth=1, alpha=0.3)
        ax.scatter(x_values, fitness_values, c='gray', s=20, alpha=0.3)

        # First Improvement path
        path_first = paths_first[start]
        for i in range(len(path_first) - 1):
            ax.annotate('', xy=(path_first[i + 1], f(path_first[i + 1])),
                        xytext=(path_first[i], f(path_first[i])),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.7))
        ax.scatter([p for p in path_first], [f(p) for p in path_first],
                   c='blue', s=100, marker='o', edgecolors='darkblue', linewidths=2,
                   label=f'First: {path_first[0]}->{path_first[-1]}', zorder=5)

        # Best Improvement path
        path_best = paths_best[start]
        for i in range(len(path_best) - 1):
            ax.annotate('', xy=(path_best[i + 1], f(path_best[i + 1])),
                        xytext=(path_best[i], f(path_best[i])),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7,
                                        linestyle='--'))
        ax.scatter([p for p in path_best], [f(p) for p in path_best],
                   c='red', s=80, marker='s', edgecolors='darkred', linewidths=2,
                   label=f'Best: {path_best[0]}->{path_best[-1]}', zorder=5)

        # Marcăm punctul de start
        ax.scatter(start, f(start), c='green', s=200, marker='*',
                   edgecolors='black', linewidths=2, zorder=10)

        ax.set_title(f'Start: x={start} ({int_to_binary(start)})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('f(x)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 32, 4))

    plt.suptitle('Exemple de Trasee de Căutare: First vs Best Improvement',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'convergence_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvat: {OUTPUT_DIR}/convergence_examples.png")


# ==============================================================================
# RAPORTARE REZULTATE
# ==============================================================================

def print_results(method_name, basins, paths):

    print("\n" + "=" * 80)
    print(f"REZULTATE PENTRU: {method_name.upper()}")
    print("=" * 80)

    # Maxime locale
    print("\nMAXIME LOCALE identificate:")
    for opt in sorted(basins.keys()):
        binary = int_to_binary(opt)
        fitness = f(opt)
        neighbors = get_neighbors(opt)
        is_local_max = all(f(opt) >= f(n) for n in neighbors)
        print(f"  x = {opt} ({binary}), f({opt}) = {fitness:.2f}")
        print(f"    Verificare maxim local: {is_local_max}")

    # Bazine de atracție
    print("\nBAZINE DE ATRACȚIE:")
    for opt in sorted(basins.keys()):
        basin = sorted(basins[opt])
        print(f"\n  Maxim local x = {opt} (f = {f(opt):.2f}):")
        print(f"    Bazin: {basin}")
        print(f"    Dimensiune bazin: {len(basin)} puncte")
        print(f"    Interval: [{min(basin)}, {max(basin)}]")

    print("\n" + "=" * 80)


def save_summary_report():

    basins_first, paths_first = analyze_basins('first')
    basins_best, paths_best = analyze_basins('best')
    local_maxima = find_local_maxima()

    report = []
    report.append("=" * 80)
    report.append("RAPORT COMPLET - ANALIZA HILL CLIMBING")
    report.append("=" * 80)
    report.append(f"\nFuncția obiectiv: f(x) = x³ - 60x² + 900x + 100")
    report.append(f"Interval: [0, 31] (reprezentare pe 5 biți)")
    report.append(f"Vecinătate: Hamming distance 1 (flip 1 bit)")

    # Maxime locale
    report.append("\n" + "=" * 80)
    report.append("MAXIME LOCALE")
    report.append("=" * 80)
    for opt in local_maxima:
        fitness_val = f(opt)  # Salvăm valoarea într-o variabilă
        report.append(f"\nx = {opt} (binary: {int_to_binary(opt)})")
        report.append(f"  f({opt}) = {fitness_val:.2f}")
        neighbors = get_neighbors(opt)
        report.append(f"  Vecini: {neighbors}")
        neighbor_fitness = [f(n) for n in neighbors]
        report.append(f"  Fitness vecini: {[f'{x:.2f}' for x in neighbor_fitness]}")

    # Comparație metode
    report.append("\n" + "=" * 80)
    report.append("COMPARAȚIE FIRST vs BEST IMPROVEMENT")
    report.append("=" * 80)

    report.append("\n--- FIRST IMPROVEMENT ---")
    for opt in sorted(basins_first.keys()):
        basin = basins_first[opt]
        fitness_val = f(opt)
        report.append(f"\nOptim x={opt} (f={fitness_val:.2f}):")
        report.append(f"  Bazin ({len(basin)} puncte): {sorted(basin)}")
        report.append(f"  Procent: {len(basin) / 32 * 100:.1f}%")

    report.append("\n--- BEST IMPROVEMENT ---")
    for opt in sorted(basins_best.keys()):
        basin = basins_best[opt]
        fitness_val = f(opt)
        report.append(f"\nOptim x={opt} (f={fitness_val:.2f}):")
        report.append(f"  Bazin ({len(basin)} puncte): {sorted(basin)}")
        report.append(f"  Procent: {len(basin) / 32 * 100:.1f}%")

    # Diferențe
    report.append("\n" + "=" * 80)
    report.append("PUNCTE CU OPTIM DIFERIT")
    report.append("=" * 80)
    diff_count = 0
    for start in range(32):
        opt_first = paths_first[start][-1]
        opt_best = paths_best[start][-1]
        if opt_first != opt_best:
            diff_count += 1
            fitness_first = f(opt_first)
            fitness_best = f(opt_best)
            report.append(f"\nStart x={start} ({int_to_binary(start)}):")
            report.append(f"  First  -> {opt_first} (f={fitness_first:.2f})")
            report.append(f"  Best   -> {opt_best} (f={fitness_best:.2f})")
            report.append(f"  Diferență: {fitness_best - fitness_first:.2f}")

    report.append(f"\nTotal puncte cu rezultate diferite: {diff_count}/32 ({diff_count / 32 * 100:.1f}%)")

    # Statistici finale
    report.append("\n" + "=" * 80)
    report.append("STATISTICI FINALE")
    report.append("=" * 80)

    success_first = len(basins_first.get(10, []))
    success_best = len(basins_best.get(10, []))

    report.append(f"\nRata de succes (găsire optim global x=10):")
    report.append(f"  First Improvement:  {success_first}/32 ({success_first / 32 * 100:.1f}%)")
    report.append(f"  Best Improvement:   {success_best}/32 ({success_best / 32 * 100:.1f}%)")
    report.append(
        f"  Îmbunătățire Best:  +{success_best - success_first} puncte ({(success_best - success_first) / 32 * 100:.1f}%)")

    # Salvare
    filename = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
    with open(filename, 'w', encoding='utf-8') as file:
        file.write('\n'.join(report))

    print(f"\n✓ Salvat: {filename}")


# ==============================================================================
# FUNCTIE PRINCIPALA
# ==============================================================================

def main():
    print("=" * 80)
    print("Analiza Algoritmului Hill Climbing")
    print("=" * 80)
    print(f"Funcția: f(x) = x³ - 60x² + 900x + 100")
    print(f"Interval: [0, 31] (reprezentare pe 5 biți)")
    print(f"Folder rezultate: {OUTPUT_DIR}/")
    print()

    # Analiza pentru ambele metode
    print("Rulare First Improvement...")
    basins_first, paths_first = analyze_basins('first')
    print_results("First Improvement", basins_first, paths_first)

    print("\nRulare Best Improvement...")
    basins_best, paths_best = analyze_basins('best')
    print_results("Best Improvement", basins_best, paths_best)

    # Generare vizualizări
    print("\n" + "=" * 80)
    print("GENERARE VIZUALIZĂRI")
    print("=" * 80)

    create_fitness_landscape()
    create_basins_comparison()
    create_transition_graph(basins_first, paths_first, "First Improvement")
    create_transition_graph(basins_best, paths_best, "Best Improvement")
    create_convergence_examples()

    # Salvare raport
    print("\n" + "=" * 80)
    print("SALVARE RAPORT")
    print("=" * 80)
    save_summary_report()

    print("\n" + "=" * 80)
    print("FINALIZAT!")
    print("=" * 80)
    print(f"Toate rezultatele au fost salvate în: {OUTPUT_DIR}/")
    print("\nFișiere generate:")
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        size = os.path.getsize(filepath)
        print(f"  - {filename} ({size:,} bytes)")


if __name__ == "__main__":
    main()