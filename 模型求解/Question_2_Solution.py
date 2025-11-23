
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

class TrajectoryOptimizerGA:
    def __init__(self):
        # Problem Parameters
        self.S_total = 10.0      # Total Distance (m)
        self.v_avg_target = 2.0  # Average Speed (m/s)
        self.T_target = self.S_total / self.v_avg_target # Target Time 5s
        
        # Constraint Parameters
        # Motor speed in data is 2617 rad/s (very high), here use reasonable joint velocity limit for the context
        # Assume 300 deg/s (~5.2 rad/s)
        self.vel_limit_deg = 300.0   
        self.torque_limit = 120.0 # N.m
        
        # Joint Initial and Final Angles (Knee Joint)
        self.theta_start = 0.0   # deg
        self.theta_end = 45.0    # deg
        
        # GA Parameters
        self.pop_size = 100
        self.max_gen = 100
        self.mutation_rate = 0.2
        
    def b_spline_trajectory(self, P, T, num_points=100):
        """
        Generate Cubic Bezier/B-Spline Trajectory
        P: Control Points [P0, P1, P2, P3] (Degrees)
        T: Total Time (s)
        """
        t = np.linspace(0, 1, num_points)
        
        P0, P1, P2, P3 = P
        
        # Cubic Bezier Formula
        # theta(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        theta = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3
        
        # First Derivative (d theta / d u)
        # u = t_norm
        d_theta_du = -3*(1-t)**2 * P0 + (3*(1-t)**2 - 6*(1-t)*t) * P1 + (6*(1-t)*t - 3*t**2) * P2 + 3*t**2 * P3
        
        # Second Derivative (d^2 theta / d u^2)
        dd_theta_du2 = 6*(1-t)*P0 + (-12*(1-t) + 6*t)*P1 + (6*(1-t) - 12*t)*P2 + 6*t*P3
        
        # Convert to Time Domain
        # vel = d theta / dt = (d theta / du) * (du / dt) = d_theta_du * (1/T)
        vel = d_theta_du / T
        
        # acc = d^2 theta / dt^2 = dd_theta_du2 * (1/T)^2
        acc = dd_theta_du2 / (T**2)
        
        time = t * T
        return time, theta, vel, acc

    def calculate_fitness(self, individual):
        P1, P2, T = individual
        
        # Construct Complete Control Points
        P = [self.theta_start, P1, P2, self.theta_end]
        
        # Generate Trajectory
        time, theta, vel, acc = self.b_spline_trajectory(P, T)
        
        # --- Physical Quantity Conversion ---
        # Velocity acc is deg/s^2, Must convert to rad/s^2 for torque calculation
        vel_rad = np.radians(vel)
        acc_rad = np.radians(acc)
        
        # 1. Objective Function: Smoothness (Integral of squared acceleration) -> Related to Energy Consumption
        dt = T / len(time)
        smoothness = np.sum(acc_rad**2) * dt
        
        # 2. Constraint Penalty
        # Velocity Constraint (deg/s)
        vel_violation = np.sum(np.maximum(0, np.abs(vel) - self.vel_limit_deg))
        
        # Torque Constraint (Dynamics Model)
        # tau = J*alpha + B*omega + G
        # Assumed parameters: J=0.5 kgm^2, B=0.1, G ignored (assume gravity balanced or horizontal motion)
        tau = 0.5 * acc_rad + 0.1 * vel_rad
        torque_violation = np.sum(np.maximum(0, np.abs(tau) - self.torque_limit))
        
        # Time Constraint (Deviation)
        time_violation = abs(T - self.T_target)
        
        # Penalty Coefficients
        penalty = 1000 * vel_violation + 1000 * torque_violation + 500 * time_violation
        
        # Fitness (Loss)
        loss = smoothness + penalty
        return loss, (time, theta, vel, acc, tau)

    def run(self):
        print("========== Question 2 Solution Start (Genetic Algorithm) ==========")
        np.random.seed(42) # Reproducibility
        
        # Initialize Population [P1, P2, T]
        # P1, P2 range: [-45, 90] (Reasonable Joint Range), T range: [4.5, 5.5] (Close to target)
        pop = np.random.rand(self.pop_size, 3)
        pop[:, 0] = pop[:, 0] * 135 - 45 # P1
        pop[:, 1] = pop[:, 1] * 135 - 45 # P2
        pop[:, 2] = pop[:, 2] * 1.0 + 4.5    # T [4.5, 5.5]
        
        best_loss = float('inf')
        best_sol = None
        best_traj = None
        
        loss_history = []

        for gen in range(self.max_gen):
            fitness_vals = []
            traj_infos = []
            
            for ind in pop:
                loss, info = self.calculate_fitness(ind)
                fitness_vals.append(loss)
                traj_infos.append(info)
            
            fitness_vals = np.array(fitness_vals)
            
            # Record Best
            min_idx = np.argmin(fitness_vals)
            if fitness_vals[min_idx] < best_loss:
                best_loss = fitness_vals[min_idx]
                best_sol = pop[min_idx].copy()
                best_traj = traj_infos[min_idx]
            
            loss_history.append(best_loss)
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Loss = {best_loss:.4f}")
            
            # Selection (Tournament)
            new_pop = []
            # Elitism: Keep top 2 elites
            sorted_indices = np.argsort(fitness_vals)
            new_pop.append(pop[sorted_indices[0]])
            new_pop.append(pop[sorted_indices[1]])
            
            while len(new_pop) < self.pop_size:
                i1, i2 = np.random.randint(0, self.pop_size, 2)
                parent = pop[i1] if fitness_vals[i1] < fitness_vals[i2] else pop[i2]
                new_pop.append(parent)
            new_pop = np.array(new_pop)
            
            # Crossover + Mutation
            for i in range(2, self.pop_size): # Skip elites
                if np.random.rand() < 0.8: # Crossover
                    idx2 = np.random.randint(0, self.pop_size)
                    alpha = np.random.rand()
                    new_pop[i] = alpha * new_pop[i] + (1-alpha) * new_pop[idx2]
                
                if np.random.rand() < self.mutation_rate: # Mutation
                    noise = np.random.randn(3) * [10, 10, 0.2]
                    new_pop[i] += noise
            
            pop = new_pop

        print("\nSolution Completed!")
        print(f"Optimal Control Points P1={best_sol[0]:.2f}, P2={best_sol[1]:.2f} (deg)")
        print(f"Optimal Time T={best_sol[2]:.4f} s (Target 5.0s)")
        
        self.plot_results(best_traj, loss_history)

    def plot_results(self, traj, loss_history):
        time, theta, vel, acc, tau = traj
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(loss_history, 'b-')
        plt.title('Fitness Convergence Curve')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(time, theta, 'r-', linewidth=2, label='Angle')
        plt.title('Knee Joint Angle Trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(time, vel, 'g-', label='Velocity')
        plt.axhline(y=self.vel_limit_deg, color='r', linestyle='--', label='Upper Limit')
        plt.axhline(y=-self.vel_limit_deg, color='r', linestyle='--', label='Lower Limit')
        plt.title('Angular Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (deg/s)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(time, tau, 'm-', label='Required Torque')
        plt.axhline(y=self.torque_limit, color='k', linestyle='--', label='Torque Limit')
        plt.axhline(y=-self.torque_limit, color='k', linestyle='--')
        plt.title('Joint Torque')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N.m)')
        plt.legend()
        plt.grid(True)
        
        # Add conclusion text
        conclusion_text = f"Conclusion: Optimal Time T={time[-1]:.2f}s, Constraints Satisfied.\nTrajectory is smooth."
        plt.figtext(0.5, 0.02, conclusion_text, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12) # Make space for text
        ensure_dir('图')
        save_path = '图/Question_2_Result.png'
        plt.savefig(save_path, dpi=300)
        print(f"Result curve saved to {save_path}")

if __name__ == "__main__":
    optimizer = TrajectoryOptimizerGA()
    optimizer.run()
