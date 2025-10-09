import numpy as np
import matplotlib.pyplot as plt

# --- FIX PENTRU MEDII FARA INTERFATA GRAFICA ---
# Se seteaza un backend non-interactiv inainte de a importa pyplot.
import matplotlib
matplotlib.use('Agg')
# --------------------------------------------------------------------

# Functia de maximizat
def objective_function(x):
    """Calculeaza valoarea functiei f(x) = x^3 - 60x^2 + 900x + 100."""
    return x**3 - 60*x**2 + 900*x + 100

# Domeniul de valori pentru x
x_values = np.arange(0, 32) # Intregi de la 0 la 31
y_values = objective_function(x_values)

# Maximele locale identificate in raport
maxima_x = [10, 21, 26]
maxima_y = [objective_function(x) for x in maxima_x]

# --- Crearea Graficului ---
plt.figure(figsize=(10, 6))

# Deseneaza curba functiei
plt.plot(x_values, y_values, '-o', markersize=4, label='Valoarea f(x)', zorder=1)

# Marcheaza maximele locale cu puncte rosii
plt.scatter(maxima_x, maxima_y, color='red', s=100, zorder=5, label='Maxime Locale')

# Adauga etichete text pentru fiecare maxim, pentru claritate
for x, y in zip(maxima_x, maxima_y):
    plt.text(x, y + 150, f'x={x}\nf(x)={y}', horizontalalignment='center', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))

# Setarea titlului si a etichetelor
plt.title('Peisajul de Fitness pentru f(x) pe Domeniul [0, 31]', fontsize=16)
plt.xlabel('Valoare x (întreg)', fontsize=12)
plt.ylabel('Valoare f(x) (Fitness)', fontsize=12)
plt.xticks(np.arange(0, 32, 2)) # Afiseaza etichete pe axa X din 2 in 2
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Salvarea figurii
output_filename = 'fitness_landscape.png'
plt.savefig(output_filename, dpi=150)

print(f"Graficul '{output_filename}' a fost generat cu succes.")
