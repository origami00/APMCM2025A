
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
possible_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC']
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
        # 问题参数
        self.S_total = 10.0      # 总距离 (m)
        self.v_avg_target = 2.0  # 平均速度 (m/s)
        self.T_target = self.S_total / self.v_avg_target # 目标时间 5s
        
        # 限制参数
        # 数据中电机转速 2617 rad/s (非常大)，这里使用题目语境下合理的关节速度限制
        # 假设 300 deg/s (~5.2 rad/s)
        self.vel_limit_deg = 300.0   
        self.torque_limit = 120.0 # N·m
        
        # 关节初始与终止角度 (膝关节)
        self.theta_start = 0.0   # deg
        self.theta_end = 45.0    # deg
        
        # GA 参数
        self.pop_size = 100
        self.max_gen = 100
        self.mutation_rate = 0.2
        
    def b_spline_trajectory(self, P, T, num_points=100):
        """
        生成 Cubic Bezier/B-Spline 轨迹
        P: 控制点 [P0, P1, P2, P3] (Degrees)
        T: 总时间 (s)
        """
        t = np.linspace(0, 1, num_points)
        
        P0, P1, P2, P3 = P
        
        # Cubic Bezier 公式
        # theta(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        theta = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3
        
        # 一阶导 (d theta / d u)
        # u = t_norm
        d_theta_du = -3*(1-t)**2 * P0 + (3*(1-t)**2 - 6*(1-t)*t) * P1 + (6*(1-t)*t - 3*t**2) * P2 + 3*t**2 * P3
        
        # 二阶导 (d^2 theta / d u^2)
        dd_theta_du2 = 6*(1-t)*P0 + (-12*(1-t) + 6*t)*P1 + (6*(1-t) - 12*t)*P2 + 6*t*P3
        
        # 转换到时间域
        # vel = d theta / dt = (d theta / du) * (du / dt) = d_theta_du * (1/T)
        vel = d_theta_du / T
        
        # acc = d^2 theta / dt^2 = dd_theta_du2 * (1/T)^2
        acc = dd_theta_du2 / (T**2)
        
        time = t * T
        return time, theta, vel, acc

    def calculate_fitness(self, individual):
        P1, P2, T = individual
        
        # 构造完整控制点
        P = [self.theta_start, P1, P2, self.theta_end]
        
        # 生成轨迹
        time, theta, vel, acc = self.b_spline_trajectory(P, T)
        
        # --- 物理量转换 ---
        # 速度 acc 是 deg/s^2, 必须转换为 rad/s^2 计算力矩
        vel_rad = np.radians(vel)
        acc_rad = np.radians(acc)
        
        # 1. 目标函数: 平滑度 (加速度平方积分) -> 能量消耗相关
        dt = T / len(time)
        smoothness = np.sum(acc_rad**2) * dt
        
        # 2. 约束惩罚
        # 速度约束 (deg/s)
        vel_violation = np.sum(np.maximum(0, np.abs(vel) - self.vel_limit_deg))
        
        # 力矩约束 (动力学模型)
        # tau = J*alpha + B*omega + G
        # 假设参数: J=0.5 kgm^2, B=0.1, G 忽略(假设重力被平衡或水平运动)
        tau = 0.5 * acc_rad + 0.1 * vel_rad
        torque_violation = np.sum(np.maximum(0, np.abs(tau) - self.torque_limit))
        
        # 时间约束 (偏差)
        time_violation = abs(T - self.T_target)
        
        # 惩罚系数
        penalty = 1000 * vel_violation + 1000 * torque_violation + 500 * time_violation
        
        # 适应度 (Loss)
        loss = smoothness + penalty
        return loss, (time, theta, vel, acc, tau)

    def run(self):
        print("========== 小问 2 求解开始 (遗传算法) ==========")
        np.random.seed(42) # 复现性
        
        # 初始化种群 [P1, P2, T]
        # P1, P2 range: [-45, 90] (合理关节范围), T range: [4.5, 5.5] (接近目标)
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
            
            # 记录最佳
            min_idx = np.argmin(fitness_vals)
            if fitness_vals[min_idx] < best_loss:
                best_loss = fitness_vals[min_idx]
                best_sol = pop[min_idx].copy()
                best_traj = traj_infos[min_idx]
            
            loss_history.append(best_loss)
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Loss = {best_loss:.4f}")
            
            # 选择 (锦标赛)
            new_pop = []
            # Elitism: 保留最好的 2 个
            sorted_indices = np.argsort(fitness_vals)
            new_pop.append(pop[sorted_indices[0]])
            new_pop.append(pop[sorted_indices[1]])
            
            while len(new_pop) < self.pop_size:
                i1, i2 = np.random.randint(0, self.pop_size, 2)
                parent = pop[i1] if fitness_vals[i1] < fitness_vals[i2] else pop[i2]
                new_pop.append(parent)
            new_pop = np.array(new_pop)
            
            # 交叉 + 变异
            for i in range(2, self.pop_size): # Skip elites
                if np.random.rand() < 0.8: # Crossover
                    idx2 = np.random.randint(0, self.pop_size)
                    alpha = np.random.rand()
                    new_pop[i] = alpha * new_pop[i] + (1-alpha) * new_pop[idx2]
                
                if np.random.rand() < self.mutation_rate: # Mutation
                    noise = np.random.randn(3) * [10, 10, 0.2]
                    new_pop[i] += noise
            
            pop = new_pop

        print("\n求解完成!")
        print(f"最优控制点 P1={best_sol[0]:.2f}, P2={best_sol[1]:.2f} (deg)")
        print(f"最优时间 T={best_sol[2]:.4f} s (目标 5.0s)")
        
        self.plot_results(best_traj, loss_history)

    def plot_results(self, traj, loss_history):
        time, theta, vel, acc, tau = traj
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(loss_history, 'b-')
        plt.title('适应度收敛曲线 (Fitness)')
        plt.xlabel('迭代代数')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(time, theta, 'r-', linewidth=2, label='角度')
        plt.title('膝关节角度轨迹 (Angle)')
        plt.xlabel('时间 (s)')
        plt.ylabel('角度 (deg)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(time, vel, 'g-', label='角速度')
        plt.axhline(y=self.vel_limit_deg, color='r', linestyle='--', label='上限')
        plt.axhline(y=-self.vel_limit_deg, color='r', linestyle='--', label='下限')
        plt.title('角速度 (Velocity)')
        plt.xlabel('时间 (s)')
        plt.ylabel('速度 (deg/s)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(time, tau, 'm-', label='所需力矩')
        plt.axhline(y=self.torque_limit, color='k', linestyle='--', label='力矩限制')
        plt.axhline(y=-self.torque_limit, color='k', linestyle='--')
        plt.title('关节力矩 (Torque)')
        plt.xlabel('时间 (s)')
        plt.ylabel('力矩 (N·m)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        ensure_dir('图')
        save_path = '图/Question_2_Result.png'
        plt.savefig(save_path, dpi=300)
        print(f"结果曲线已保存至 {save_path}")

if __name__ == "__main__":
    optimizer = TrajectoryOptimizerGA()
    optimizer.run()
