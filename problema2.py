import numpy as np

class DualPhaseSimplex:
    def __init__(self):
        self.tableau = None
        self.basic_vars = None
        self.artificial_vars = None
        
    def print_tableau(self, phase, iteration):
        print(f"\n--- Fase {phase}, Iteración {iteration} ---")
        print("Tableau:")
        for row in self.tableau:
            print(" ".join(f"{val:8.3f}" for val in row))
        print(f"Variables básicas: {self.basic_vars}")
        
    def initialize_phase1(self, A, b, c, constraint_types):
        """Inicializa la Fase I"""
        m, n = A.shape
        artificial_indices = [i for i, t in enumerate(constraint_types) if t in ['=', '>=']]
        num_artificial = len(artificial_indices)
        total_vars = n + len([t for t in constraint_types if t in ['<=', '>=']]) + num_artificial
        
        self.tableau = np.zeros((m + 1, total_vars + 1))
        
        # Copiar A
        self.tableau[:m, :n] = A
        
        # Agregar holguras/excesos
        slack_col = n
        for i, t in enumerate(constraint_types):
            if t == '<=':
                self.tableau[i, slack_col] = 1
                slack_col += 1
            elif t == '>=':
                self.tableau[i, slack_col] = -1
                slack_col += 1
        
        # Agregar artificiales
        for k, i in enumerate(artificial_indices):
            self.tableau[i, slack_col + k] = 1
            self.tableau[-1, slack_col + k] = 1
        
        self.tableau[:m, -1] = b
        
        self.basic_vars = list(range(slack_col, slack_col + num_artificial))
        self.artificial_vars = list(self.basic_vars)
        
        # Ajustar fila objetivo
        for i in range(m):
            if i in artificial_indices:
                self.tableau[-1, :] -= self.tableau[i, :]
                
        return slack_col, num_artificial
    
    def pivot(self, entering, leaving):
        pivot_element = self.tableau[leaving, entering]
        self.tableau[leaving, :] /= pivot_element
        for i in range(self.tableau.shape[0]):
            if i != leaving:
                self.tableau[i, :] -= self.tableau[i, entering] * self.tableau[leaving, :]
        if leaving < len(self.basic_vars):
            self.basic_vars[leaving] = entering
    
    def simplex_iteration(self, phase):
        m = self.tableau.shape[0] - 1
        objective_row = self.tableau[-1, :-1]
        entering = np.argmin(objective_row)
        min_reduced_cost = objective_row[entering]
        if min_reduced_cost >= -1e-10:
            return True, None
        
        ratios = []
        for i in range(m):
            if self.tableau[i, entering] > 1e-10:
                ratios.append((self.tableau[i, -1] / self.tableau[i, entering], i))
            else:
                ratios.append((float('inf'), i))
        
        ratios = sorted(ratios, key=lambda x: x[0])
        for ratio, idx in ratios:
            if ratio != float('inf') and ratio >= 0:
                return False, (entering, idx)
        return False, "Problema ilimitado"
    
    def solve_phase1(self, A, b, c, constraint_types):
        print("=== FASE I ===")
        slack_col, num_artificial = self.initialize_phase1(A, b, c, constraint_types)
        iteration = 0
        self.print_tableau(1, iteration)
        
        while True:
            optimal, result = self.simplex_iteration(1)
            if optimal:
                if abs(self.tableau[-1, -1]) > 1e-8:
                    return False, "Problema infactible"
                else:
                    break
            elif result == "Problema ilimitado":
                return False, "Problema ilimitado en Fase I"
            else:
                entering, leaving = result
                self.pivot(entering, leaving)
                iteration += 1
                self.print_tableau(1, iteration)
        
        return True, (slack_col, num_artificial)
    
    def initialize_phase2(self, original_c, n_vars, slack_col, num_artificial):
        print("\n=== FASE II ===")
        keep_cols = [j for j in range(self.tableau.shape[1]) 
                     if j not in range(slack_col, slack_col + num_artificial)]
        self.tableau = self.tableau[:, keep_cols]
        self.basic_vars = [b for b in self.basic_vars if b < self.tableau.shape[1] - 1]
        
        # Nueva función objetivo
        self.tableau[-1, :] = 0
        self.tableau[-1, :n_vars] = -np.array(original_c)
        
        # Ajustar por las variables básicas actuales
        for i, b in enumerate(self.basic_vars):
            self.tableau[-1, :] -= self.tableau[-1, b] * self.tableau[i, :]
    
    def solve_phase2(self):
        iteration = 0
        self.print_tableau(2, iteration)
        while True:
            optimal, result = self.simplex_iteration(2)
            if optimal:
                break
            elif result == "Problema ilimitado":
                return False, "Problema ilimitado", None
            else:
                entering, leaving = result
                self.pivot(entering, leaving)
                iteration += 1
                self.print_tableau(2, iteration)
        
        solution = np.zeros(self.tableau.shape[1] - 1)
        for i, b in enumerate(self.basic_vars):
            if b < len(solution):
                solution[b] = self.tableau[i, -1]
        
        optimal_value = -self.tableau[-1, -1]
        return True, solution, optimal_value


# -------------------
# BLOQUE PRINCIPAL
# -------------------
def solve_problem2():
    print("PROBLEMA 2: Método Simplex Dual Phase")
    print("Minimizar Z = 5x1 - 4x2 + 3x3")
    print("Sujeto a:")
    print("2x1 + x2 - x3 = 10")
    print("x1 - 3x2 + 2x3 ≥ 5")
    print("x1 + x2 + x3 ≤ 15")
    print("x1, x2, x3 ≥ 0")
    
    original_c = [-5, 4, -3]
    A = np.array([
        [2, 1, -1],
        [1, -3, 2],
        [1, 1, 1]
    ])
    b = np.array([10, 5, 15])
    constraint_types = ['=', '>=', '<=']

    solver = DualPhaseSimplex()
    success, data = solver.solve_phase1(A, b, original_c, constraint_types)
    
    if not success:
        print("Fase I falló:", data)
        return
    
    slack_col, num_artificial = data
    solver.initialize_phase2(original_c, len(original_c), slack_col, num_artificial)
    success, solution, optimal_value = solver.solve_phase2()
    
    if success:
        print("\n*** SOLUCIÓN ÓPTIMA ENCONTRADA ***")
        print(f"Valor óptimo: {optimal_value:.3f}")
        print(f"x1 = {solution[0]:.3f}, x2 = {solution[1]:.3f}, x3 = {solution[2]:.3f}")
        print("\nVerificación de restricciones:")
        print(f"2x1 + x2 - x3 = {2*solution[0] + solution[1] - solution[2]:.3f} (debe ser 10)")
        print(f"x1 - 3x2 + 2x3 = {solution[0] - 3*solution[1] + 2*solution[2]:.3f} (≥ 5)")
        print(f"x1 + x2 + x3 = {sum(solution):.3f} (≤ 15)")
    else:
        print("Fase II falló:", solution)

if __name__ == "__main__":
    solve_problem2()
