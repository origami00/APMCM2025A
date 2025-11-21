
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TrajectoryOptimizerGA:
    def __init__(self):
        # 问题参数
        self.S_total = 10.0      # 总距离 (m)
        self.v_avg_target = 2.0  # 平均速度 (m/s)
        self.T_target = self.S_total / self.v_avg_target # 目标时间 5s
        
        self.vel_limit = 10.0    # 角速度限制 (deg/s)
        self.torque_limit = 8.0  # 力矩限制 (N·m)
        
        # 关节初始与终止角度 (假设)
        self.theta_start = 0.0
        self.theta_end = 45.0    # 假设行走一步膝关节弯曲到45度
        
        # GA 参数
        self.pop_size = 50
        self.max_gen = 50
        self.mutation_rate = 0.1
        
    def b_spline_trajectory(self, P, T, num_points=100):
        """
        生成 B 样条轨迹
        P: 控制点 [P0, P1, P2, P3]
        T: 总时间
        """
        # B样条阶数 k=3 (4个控制点)
        k = 3
        # 节点向量 (Clamped B-Spline)
        t = np.linspace(0, 1, num_points)
        # 简单的 Cubic Bezier / B-Spline 实现
        # 这里使用 explicit formula for cubic Bezier (也是一种 B-spline 特例) 方便求导
        # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        
        P0, P1, P2, P3 = P
        
        theta = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3
        
        # 计算速度 (链式法则 dtheta/dt = dtheta/du * du/dt, du/dt = 1/T)
        d_theta_du = -3*(1-t)**2 * P0 + (3*(1-t)**2 - 6*(1-t)*t) * P1 + (6*(1-t)*t - 3*t**2) * P2 + 3*t**2 * P3
        vel = d_theta_du / T
        
        # 计算加速度
        dd_theta_du2 = 6*(1-t)*P0 + (-12*(1-t) + 6*t)*P1 + (6*(1-t) - 12*t)*P2 + 6*t*P3
        acc = dd_theta_du2 / (T**2)
        
        time = t * T
        return time, theta, vel, acc

    def calculate_fitness(self, individual):
        P1, P2, T = individual
        
        # 构造完整控制点
        P = [self.theta_start, P1, P2, self.theta_end]
        
        # 生成轨迹
        time, theta, vel, acc = self.b_spline_trajectory(P, T)
        
        # 1. 目标函数: 平滑度 (加速度平方积分)
        # J = integral(acc^2) dt approx sum(acc^2) * dt
        dt = T / len(time)
        smoothness = np.sum(acc**2) * dt
        
        # 2. 约束惩罚
        # 速度约束
        vel_violation = np.sum(np.maximum(0, np.abs(vel) - self.vel_limit))
        
        # 力矩约束 (简化动力学: tau = J*acc + B*vel + G)
        # 假设 J=0.5, B=0.1, G=0 (简化)
        tau = 0.5 * acc + 0.1 * vel
        torque_violation = np.sum(np.maximum(0, np.abs(tau) - self.torque_limit))
        
        # 时间约束 (soft penalty for deviating from 5s)
        time_violation = abs(T - self.T_target)
        
        # 总惩罚
        penalty = 1000 * vel_violation + 1000 * torque_violation + 100 * time_violation
        
        # 适应度 (越小越好 -> 取倒数或负数，这里直接返回 loss 越小越好)
        loss = smoothness + penalty
        return loss, (time, theta, vel, acc, tau)

    def run(self):
        print("========== 小问 2 求解开始 (遗传算法) ==========")
        
        # 初始化种群 [P1, P2, T]
        # P1, P2 range: [-90, 90], T range: [4, 6]
        pop = np.random.rand(self.pop_size, 3)
        pop[:, 0] = pop[:, 0] * 180 - 90 # P1
        pop[:, 1] = pop[:, 1] * 180 - 90 # P2
        pop[:, 2] = pop[:, 2] * 2 + 4    # T [4, 6]
        
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
            
            # 选择 (Tournament Selection)
            new_pop = []
            for _ in range(self.pop_size):
                i1, i2 = np.random.randint(0, self.pop_size, 2)
                parent = pop[i1] if fitness_vals[i1] < fitness_vals[i2] else pop[i2]
                new_pop.append(parent)
            new_pop = np.array(new_pop)
            
            # 交叉 + 变异
            for i in range(self.pop_size):
                if np.random.rand() < 0.8: # Crossover
                    idx2 = np.random.randint(0, self.pop_size)
                    alpha = np.random.rand()
                    new_pop[i] = alpha * new_pop[i] + (1-alpha) * new_pop[idx2]
                
                if np.random.rand() < self.mutation_rate: # Mutation
                    noise = np.random.randn(3) * [10, 10, 0.5]
                    new_pop[i] += noise
            
            pop = new_pop

        print("\n求解完成!")
        print(f"最优控制点 P1={best_sol[0]:.2f}, P2={best_sol[1]:.2f}")
        print(f"最优时间 T={best_sol[2]:.2f} s")
        
        # 可视化
        self.plot_results(best_traj, loss_history)

    def plot_results(self, traj, loss_history):
        time, theta, vel, acc, tau = traj
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(loss_history)
        plt.title('适应度收敛曲线')
        plt.xlabel('迭代代数')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(time, theta, 'b-', label='角度')
        plt.title('膝关节角度变化')
        plt.xlabel('时间 (s)')
        plt.ylabel('角度 (deg)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(time, vel, 'g-', label='角速度')
        plt.plot([0, time[-1]], [self.vel_limit, self.vel_limit], 'r--', label='上限')
        plt.plot([0, time[-1]], [-self.vel_limit, -self.vel_limit], 'r--', label='下限')
        plt.title('角速度变化')
        plt.xlabel('时间 (s)')
        plt.ylabel('速度 (deg/s)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(time, tau, 'm-', label='力矩')
        plt.plot([0, time[-1]], [self.torque_limit, self.torque_limit], 'r--', label='上限')
        plt.plot([0, time[-1]], [-self.torque_limit, -self.torque_limit], 'r--', label='下限')
        plt.title('关节力矩变化')
        plt.xlabel('时间 (s)')
        plt.ylabel('力矩 (N·m)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('模型求解/Question_2_Result.png')
        print("结果曲线已保存至 Question_2_Result.png")

if __name__ == "__main__":
    optimizer = TrajectoryOptimizerGA()
    optimizer.run()

