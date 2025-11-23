
import numpy as np
import matplotlib.pyplot as plt
import os

# Set fonts
possible_fonts = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
for font in possible_fonts:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except:
        continue

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class NSGA2_Optimizer:
    def __init__(self):
        # Optimization Variable Ranges
        # x1: omega_1 (deg/s) - Action 1 Velocity
        # x2: T_2     (s)     - Action 2 Time
        # x3: omega_3 (deg/s) - Action 3 Velocity
        self.bounds = np.array([
            [10.0, 120.0],  # omega_1 (Ref Q1, range broadened)
            [2.0, 8.0],     # T_2 (Ref Q2, target 5s)
            [30.0, 180.0]   # omega_3 (Ref Q3, T=4s -> 90deg/s)
        ])
        
        # Algorithm Parameters
        self.pop_size = 100
        self.max_gen = 50
        
    def evaluate(self, x):
        """
        Evaluate Individual
        x: [omega_1, T_2, omega_3]
        Returns: [Energy (J), MaxTime (s)] (Min, Min)
        """
        w1_deg, t2, w3_deg = x
        
        # --- Objective 1: Cycle Time ---
        # Assume three actions serial or partial parallel. Problem implies different phases.
        # Phase 1: Extension+Rotation. Total angle approx 90 deg. t1 = 90 / w1
        t1 = 90.0 / w1_deg
        
        # Phase 2: Walking. t2 = t2 (Direct optim variable)
        
        # Phase 3: Circling. One cycle 360 deg. t3 = 360 / w3
        t3 = 360.0 / w3_deg
        
        # Total Time (Assume Serial)
        cycle_time = t1 + t2 + t3
        
        # --- Objective 2: Total Energy ---
        # Energy = Power * Time
        # Power Model: P(w) = P_static + k1*w + k2*w^2
        
        def calc_energy_segment(w_deg, t_duration, torque_load_factor):
            w_rad = np.radians(w_deg)
            # Base Loss (Control circuits etc.)
            p_base = 5.0 
            # Mechanical Loss (Friction ~ w)
            p_mech = 2.0 * w_rad 
            # Copper Loss (I^2*R ~ T^2 ~ (Load + acc)^2). Simplified as related to w^2 (viscous) or constant (gravity)
            # Assume constant gravity load dominant: P_load = T * w
            p_load = torque_load_factor * 20.0 * w_rad # 20Nm avg load
            # Extra Copper Loss Term
            p_cu = 1.0 * (w_rad ** 2)
            
            power = p_base + p_mech + p_load + p_cu
            return power * t_duration

        # Calculate energy for each phase
        e1 = calc_energy_segment(w1_deg, t1, 1.5) # High torque during extension
        
        # Phase 2 Velocity Estimation: Distance=10m. v_avg = 10/t2. 
        # Leg Swing Angular Velocity w2 ~ v_avg * Const. 
        v_avg = 10.0 / t2
        w2_deg = v_avg * 30.0 # Rough Estimate
        e2 = calc_energy_segment(w2_deg, t2, 1.0)
        
        e3 = calc_energy_segment(w3_deg, t3, 0.8) # Low load during circling
        
        total_energy = e1 + e2 + e3
        
        return np.array([total_energy, cycle_time])

    def fast_non_dominated_sort(self, population_objs):
        pop_size = len(population_objs)
        domination_count = np.zeros(pop_size)
        dominated_solutions = [[] for _ in range(pop_size)]
        ranks = np.zeros(pop_size)
        
        fronts = [[]]
        
        for p in range(pop_size):
            for q in range(pop_size):
                p_obj = population_objs[p]
                q_obj = population_objs[q]
                
                # Check domination: p dominates q if p <= q and p < q in at least one
                if np.all(p_obj <= q_obj) and np.any(p_obj < q_obj):
                    dominated_solutions[p].append(q)
                elif np.all(q_obj <= p_obj) and np.any(q_obj < p_obj):
                    domination_count[p] += 1
            
            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)
                
        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            if len(next_front) > 0:
                fronts.append(next_front)
            
        return fronts, ranks

    def crowding_distance(self, population_objs, front):
        l = len(front)
        distances = np.zeros(l)
        if l == 0: return distances
        
        num_obj = population_objs.shape[1]
        
        for m in range(num_obj):
            # Sort front by objective m
            sorted_indices = np.argsort(population_objs[front, m])
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_min = population_objs[front[sorted_indices[0]], m]
            obj_max = population_objs[front[sorted_indices[-1]], m]
            obj_range = obj_max - obj_min
            
            if obj_range == 0: obj_range = 1.0
            
            for i in range(1, l-1):
                distances[sorted_indices[i]] += (
                    population_objs[front[sorted_indices[i+1]], m] - 
                    population_objs[front[sorted_indices[i-1]], m]
                ) / obj_range
                
        return distances

    def run(self):
        print("========== Question 4 Solution Start (NSGA-II Multi-objective Optimization) ==========")
        np.random.seed(1)
        
        # 1. Initialization
        pop = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, 3))
        
        for gen in range(self.max_gen):
            objs = np.array([self.evaluate(ind) for ind in pop])
            
            fronts, _ = self.fast_non_dominated_sort(objs)
            
            # Generate Offspring (Simple Mutation + Crossover)
            offspring = pop.copy()
            # Mutation
            mask = np.random.rand(*pop.shape) < 0.3
            noise = np.random.normal(0, 5.0, pop.shape)
            offspring[mask] += noise[mask]
            # Clip
            for i in range(3):
                offspring[:, i] = np.clip(offspring[:, i], self.bounds[i,0], self.bounds[i,1])
            
            # Merge
            combined_pop = np.vstack((pop, offspring))
            combined_objs = np.array([self.evaluate(ind) for ind in combined_pop])
            
            # Sort
            fronts, _ = self.fast_non_dominated_sort(combined_objs)
            
            # Select
            new_pop_indices = []
            for front in fronts:
                if len(new_pop_indices) + len(front) <= self.pop_size:
                    new_pop_indices.extend(front)
                else:
                    dists = self.crowding_distance(combined_objs, front)
                    # Sort front by distance descending
                    sorted_front_indices = np.argsort(dists)[::-1]
                    needed = self.pop_size - len(new_pop_indices)
                    # Get actual indices from front
                    best_in_front = [front[i] for i in sorted_front_indices[:needed]]
                    new_pop_indices.extend(best_in_front)
                    break
            
            pop = combined_pop[new_pop_indices]
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Pareto Front Size = {len(fronts[0])}")
        
        # Final Results
        final_objs = np.array([self.evaluate(ind) for ind in pop])
        fronts, _ = self.fast_non_dominated_sort(final_objs)
        pareto_front = final_objs[fronts[0]]
        
        print("\nSolution Completed!")
        print(f"Number of Pareto Front Solutions: {len(pareto_front)}")
        
        self.plot_results(pareto_front)

    def plot_results(self, pareto_front):
        # Sort for plotting line
        sorted_indices = np.argsort(pareto_front[:, 1])
        pareto_front = pareto_front[sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.plot(pareto_front[:, 1], pareto_front[:, 0], 'r-o', label='Pareto Front')
        
        # Knee point
        # Normalize
        min_vals = np.min(pareto_front, axis=0)
        max_vals = np.max(pareto_front, axis=0)
        norm = (pareto_front - min_vals) / (max_vals - min_vals + 1e-6)
        # Closest to (0,0) in normalized space (ideal point)
        dists = np.sum(norm**2, axis=1)
        knee_idx = np.argmin(dists)
        knee_point = pareto_front[knee_idx]
        
        plt.scatter(knee_point[1], knee_point[0], c='blue', s=150, marker='*', zorder=5, label='Recommendation (Knee)')
        
        plt.title('Multi-objective Optimization: Energy vs Time')
        plt.xlabel('Total Cycle Time (s)')
        plt.ylabel('Total Energy Consumption (J)')
        plt.grid(True)
        plt.legend()
        
        # Add conclusion text
        conclusion_text = f"Conclusion: Recommended (Knee Point) - Time {knee_point[1]:.2f}s, Energy {knee_point[0]:.2f}J.\nTrade-off between efficiency and energy (Pareto Optimal)."
        plt.figtext(0.5, 0.02, conclusion_text, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.subplots_adjust(bottom=0.15)
        
        ensure_dir('图')
        save_path = '图/Question_4_Result.png'
        plt.savefig(save_path, dpi=300)
        print(f"Recommended Solution: Time={knee_point[1]:.2f}s, Energy={knee_point[0]:.2f}J")
        print(f"Result plot saved to {save_path}")

if __name__ == "__main__":
    optimizer = NSGA2_Optimizer()
    optimizer.run()
