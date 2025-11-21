
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NSGA2_Optimizer:
    def __init__(self):
        # 优化变量范围
        # x1: omega_1 (0, 5]
        # x2: T_2     (4, 6]
        # x3: omega_3 (0, 10]
        self.bounds = np.array([
            [0.1, 5.0],
            [4.0, 6.0],
            [0.1, 10.0]
        ])
        
        # 算法参数
        self.pop_size = 100
        self.max_gen = 100
        
    def evaluate(self, x):
        """
        评估个体
        x: [omega_1, T_2, omega_3]
        Returns: [Energy, MaxTime]
        """
        w1, t2, w3 = x
        
        # 1. 计算时间目标 (Max Cycle Time)
        # t1 = Angle / w1. 假设 Q1 动作总角度 90度 (60+30, 粗略)
        t1 = 90.0 / w1
        # t2 = t2
        # t3 = 360 / w3 (画圆一周) (w3 is deg/s here for consistency)
        t3 = 360.0 / w3
        
        max_time = max(t1, t2, t3)
        
        # 2. 计算能耗目标 (Energy)
        # 这是一个简化模型，基于 w 和 t
        # Power ~ k1 * w + k2 * w^2 (铜损) + k3 * w^4 (铁损)
        # Energy = Power * t
        
        # 损耗系数 (假设)
        k_fric = 2.0
        k_cu = 0.5
        k_fe = 0.001
        
        def calc_energy_step(w, t):
            # 简单的损耗模型
            p_mech = k_fric * w
            p_cu = k_cu * (w ** 2) # 假设力矩与速度成正比或常数，这里简化为与速度平方相关
            p_fe = k_fe * (w ** 4)
            return (p_mech + p_cu + p_fe) * t
            
        e1 = calc_energy_step(w1, t1)
        e2 = calc_energy_step(10.0, t2) # 假设 Q2 平均速度 10
        e3 = calc_energy_step(w3, t3)
        
        total_energy = e1 + e2 + e3
        
        return np.array([total_energy, max_time])

    def non_dominated_sort(self, population_objs):
        pop_size = len(population_objs)
        domination_count = np.zeros(pop_size)
        dominated_solutions = [[] for _ in range(pop_size)]
        ranks = np.zeros(pop_size)
        
        fronts = [[]]
        
        for p in range(pop_size):
            for q in range(pop_size):
                # Check if p dominates q
                # Min objectives
                p_obj = population_objs[p]
                q_obj = population_objs[q]
                
                if np.all(p_obj <= q_obj) and np.any(p_obj < q_obj):
                    dominated_solutions[p].append(q)
                elif np.all(q_obj <= p_obj) and np.any(q_obj < p_obj):
                    domination_count[p] += 1
            
            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)
                
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
            
        return fronts[:-1], ranks # Remove last empty front

    def crowding_distance(self, population_objs, front):
        l = len(front)
        distances = np.zeros(l)
        
        if l == 0: return distances
        
        num_obj = population_objs.shape[1]
        
        for m in range(num_obj):
            # Sort by objective m
            sorted_indices = sorted(range(l), key=lambda x: population_objs[front[x]][m])
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_range = population_objs[front[sorted_indices[-1]]][m] - population_objs[front[sorted_indices[0]]][m]
            if obj_range == 0: obj_range = 1.0
            
            for i in range(1, l-1):
                distances[sorted_indices[i]] += (population_objs[front[sorted_indices[i+1]]][m] - population_objs[front[sorted_indices[i-1]]][m]) / obj_range
                
        return distances

    def run(self):
        print("========== 小问 4 求解开始 (NSGA-II 多目标优化) ==========")
        
        # 1. 初始化种群
        pop = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, 3))
        
        for gen in range(self.max_gen):
            # 评估
            objs = np.array([self.evaluate(ind) for ind in pop])
            
            # 非支配排序
            fronts, ranks = self.non_dominated_sort(objs)
            
            # 选择、交叉、变异 (简单版: 随机生成子代并混合)
            # 这里简化实现：生成新随机种群，与老种群合并，然后保留前N个
            offspring = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, 3))
            # 加上一些基于父代的变异
            mask = np.random.rand(self.pop_size, 3) < 0.5
            offspring[mask] = pop[mask] + np.random.normal(0, 0.5, np.sum(mask))
            # Clip
            for i in range(3):
                offspring[:, i] = np.clip(offspring[:, i], self.bounds[i,0], self.bounds[i,1])
            
            # 合并
            combined_pop = np.vstack((pop, offspring))
            combined_objs = np.array([self.evaluate(ind) for ind in combined_pop])
            
            # 重新排序
            fronts, ranks = self.non_dominated_sort(combined_objs)
            
            # 筛选下一代
            new_pop_indices = []
            for front in fronts:
                if len(new_pop_indices) + len(front) <= self.pop_size:
                    new_pop_indices.extend(front)
                else:
                    # 计算拥挤度并截断
                    dists = self.crowding_distance(combined_objs, front)
                    # Sort by distance desc
                    sorted_front = [x for _, x in sorted(zip(dists, front), key=lambda pair: pair[0], reverse=True)]
                    needed = self.pop_size - len(new_pop_indices)
                    new_pop_indices.extend(sorted_front[:needed])
                    break
            
            pop = combined_pop[new_pop_indices]
            if gen % 20 == 0:
                print(f"Gen {gen}: Pareto Front Size = {len(fronts[0])}")
        
        # 最终结果
        final_objs = np.array([self.evaluate(ind) for ind in pop])
        fronts, _ = self.non_dominated_sort(final_objs)
        pareto_front = final_objs[fronts[0]]
        
        print("\n求解完成!")
        print(f"Pareto 前沿解数量: {len(pareto_front)}")
        
        # 可视化
        self.plot_results(pareto_front)

    def plot_results(self, pareto_front):
        plt.figure(figsize=(10, 6))
        plt.scatter(pareto_front[:, 1], pareto_front[:, 0], c='red', label='Pareto Front')
        
        # 找一个 Knee Point (简单的: 归一化后距离原点最近)
        # Normalize
        min_vals = np.min(pareto_front, axis=0)
        max_vals = np.max(pareto_front, axis=0)
        norm_front = (pareto_front - min_vals) / (max_vals - min_vals + 1e-6)
        dists = np.sum(norm_front**2, axis=1)
        knee_idx = np.argmin(dists)
        knee_point = pareto_front[knee_idx]
        
        plt.scatter(knee_point[1], knee_point[0], c='blue', s=100, marker='*', label='推荐解 (Knee Point)')
        
        plt.title('小问 4: 多目标优化 Pareto 前沿')
        plt.xlabel('最大周期时间 (s) - [效率]')
        plt.ylabel('总能耗 (J) - [节能]')
        plt.grid(True)
        plt.legend()
        
        plt.savefig('模型求解/Question_4_Result.png')
        print(f"推荐解: 时间={knee_point[1]:.2f}s, 能耗={knee_point[0]:.2f}J")
        print("结果图已保存至 Question_4_Result.png")

if __name__ == "__main__":
    optimizer = NSGA2_Optimizer()
    optimizer.run()

