import numpy as np
import matplotlib.pyplot as plt

class Simplex:
    def __init__(self, c, A, b):
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        self.m, self.n = self.A.shape  # Corregido: self.A.shape
        
    def solve(self):
        # Tabla inicial: [A | I | b] y función objetivo
        tableau = np.zeros((self.m + 1, self.n + self.m + 1))
        tableau[:-1, :self.n] = self.A
        tableau[:-1, self.n:self.n + self.m] = np.eye(self.m)
        tableau[:-1, -1] = self.b
        tableau[-1, :self.n] = -self.c
        
        basic_vars = list(range(self.n, self.n + self.m))
        iterations = 0
        
        print("=== MÉTODO SIMPLEX ===")
        print("Tabla inicial:")
        self.print_tableau(tableau, basic_vars)
        
        while np.any(tableau[-1, :-1] < -1e-10):
            # Variable de entrada (más negativo)
            entering = np.argmin(tableau[-1, :-1])
            
            # Variable de salida (mínima razón positiva)
            ratios = []
            for i in range(self.m):
                if tableau[i, entering] > 1e-10:
                    ratios.append((tableau[i, -1] / tableau[i, entering], i))
                else:
                    ratios.append((np.inf, i))
            
            if all(r[0] == np.inf for r in ratios):
                raise ValueError("Problema ilimitado")
            
            leaving_idx = min(ratios)[1]
            
            # Pivoteo
            pivot = tableau[leaving_idx, entering]
            tableau[leaving_idx] /= pivot
            
            for i in range(self.m + 1):
                if i != leaving_idx:
                    tableau[i] -= tableau[i, entering] * tableau[leaving_idx]
            
            basic_vars[leaving_idx] = entering
            iterations += 1
            
            print(f"\nIteración {iterations}:")
            self.print_tableau(tableau, basic_vars)
        
        # Extraer solución
        solution = np.zeros(self.n)
        for i, var in enumerate(basic_vars):
            if var < self.n:
                solution[var] = tableau[i, -1]
        
        optimal = tableau[-1, -1]
        
        print(f"\nSolución óptima en {iterations} iteraciones")
        print(f"x = {solution}")
        print(f"Z = {optimal:.2f}")
        
        return solution, optimal, tableau, basic_vars
    
    def print_tableau(self, tableau, basic_vars):
        headers = [f"x{i+1}" for i in range(self.n)] + [f"s{i+1}" for i in range(self.m)] + ["b"]
        print("   " + "".join(f"{h:>8}" for h in headers))
        
        for i in range(self.m):
            var_type = "s" if basic_vars[i] >= self.n else "x"
            var_num = basic_vars[i] + 1 if basic_vars[i] < self.n else basic_vars[i] - self.n + 1
            print(f"{var_type}{var_num} " + "".join(f"{val:8.2f}" for val in tableau[i]))
        
        print(" Z " + "".join(f"{val:8.2f}" for val in tableau[-1]))

def sensitivity_analysis(tableau, basic_vars, n_vars, m):
    print("\n=== ANÁLISIS DE SENSIBILIDAD ===")
    
    # Precios sombra (coeficientes de variables de holgura en fila Z)
    shadow_prices = tableau[-1, n_vars:n_vars + m]
    print("Precios sombra:")
    for i, price in enumerate(shadow_prices):
        print(f"  Restricción {i+1}: {price:.4f}")
    
    # Costos reducidos
    print("\nCostos reducidos de variables no básicas:")
    for j in range(n_vars):
        if j not in basic_vars:
            reduced_cost = tableau[-1, j]
            print(f"  x{j+1}: {reduced_cost:.4f}")

def main():
    # Problema 1: Maximizar Z = 3x1 + 2x2 + 5x3
    c = [3, 2, 5]
    A = [[1, 1, 1],
         [2, 1, 1],
         [1, 4, 2]]
    b = [100, 150, 80]
    
    print("PROBLEMA 1:")
    print("Maximizar Z = 3x1 + 2x2 + 5x3")
    print("Sujeto a:")
    print("  x1 + x2 + x3 ≤ 100")
    print("  2x1 + x2 + x3 ≤ 150")
    print("   x1 + 4x2 + 2x3 ≤ 80")
    print("  x1, x2, x3 ≥ 0\n")
    
    # Resolver
    simplex = Simplex(c, A, b)
    solution, optimal, tableau, basic_vars = simplex.solve()
    
    # Verificar restricciones
    print("\nVERIFICACIÓN DE RESTRICCIONES:")
    for i in range(3):
        lhs = sum(A[i][j] * solution[j] for j in range(3))
        status = "✓" if lhs <= b[i] + 1e-6 else "✗"
        print(f"  Restricción {i+1}: {lhs:.1f} ≤ {b[i]} {status}")
    
    # Análisis de sensibilidad
    sensitivity_analysis(tableau, basic_vars, 3, 3)
    
    print("\nINTERPRETACIÓN:")
    print("Los precios sombra indican cuánto mejoraría Z si relajamos cada restricción")
    print("Los costos reducidos muestran cuánto empeoraría Z si forzamos variables cero a ser positivas")

if __name__ == "__main__":
    main()
    
# Funciones de las restricciones
x1 = np.linspace(0, 100, 200)

# Restricción 1: x1 + x2 + x3 <= 100 (solo en 2D: x1 y x2)
x2_1 = 100 - x1

# Restricción 2: 2x1 + x2 + x3 <= 150
x2_2 = 150 - 2 * x1

# Restricción 3: x1 + 4x2 + 2x3 <= 80
x2_3 = (80 - x1) / 4

# Graficar
plt.figure(figsize=(8,8))
plt.plot(x1, x2_1, label="x1 + x2 <= 100")
plt.plot(x1, x2_2, label="2x1 + x2 <= 150")
plt.plot(x1, x2_3, label="x1 + 4x2 <= 80")
plt.xlim((0, 100))
plt.ylim((0, 100))
plt.xlabel("x1")
plt.ylabel("x2")

plt.fill_between(x1, 0, np.minimum(np.minimum(x2_1, x2_2), x2_3), where=(x1 >= 0), alpha=0.3, label="Región factible")
plt.legend()
plt.title("Región factible del problema de optimización")
plt.grid(True)
plt.show()