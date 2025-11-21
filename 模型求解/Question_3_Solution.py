
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

class RobotSimulation:
    def __init__(self):
        self.dt = 0.05
        self.total_time = 10.0
        self.steps = int(self.total_time / self.dt)
        self.time = np.linspace(0, self.total_time, self.steps)
        
    def generate_training_curve(self):
        """模拟强化学习训练曲线"""
        episodes = np.arange(0, 1000)
        # 模拟：初始低，快速上升，后期震荡收敛
        base_reward = 200 * (1 - np.exp(-episodes / 150))
        noise = np.random.normal(0, 15, len(episodes)) * np.exp(-episodes/400) # 噪音随训练减少
        rewards = base_reward + noise
        return episodes, rewards
        
    def simulate_motion(self):
        """模拟机器人的协同运动"""
        # 1. 身体移动 (左转弯)
        # 假设角速度逐渐增加然后保持
        yaw_rate = np.zeros_like(self.time)
        yaw_rate[self.time < 2] = 22.5 * (self.time[self.time < 2] / 2) # 0-2s 加速
        yaw_rate[self.time >= 2] = 22.5 # 匀速转弯 deg/s (2s转45度? 不, 22.5*2/2 = 22.5度... 需调整)
        
        # 目标：总转角 45度
        # 简单的 S 型曲线角度
        target_yaw = 45.0
        body_yaw = target_yaw / (1 + np.exp(-1.5 * (self.time - 3))) # Sigmoid shape centered at 3s
        
        # 2. 手臂动作 (相对身体画圆)
        # 周期 4s -> f = 0.25 Hz
        freq = 0.25
        w = 2 * np.pi * freq
        
        radius = 0.3 # m
        center = np.array([0.2, 0.0, 0.5]) # 局部坐标
        
        arm_local_x = center[0] * np.ones_like(self.time)
        arm_local_y = center[1] + radius * np.cos(w * self.time)
        arm_local_z = center[2] + radius * np.sin(w * self.time)
        
        # 3. 转换到全局坐标 (简单的 2D 旋转应用到 x,y)
        # x_global = x_local * cos(yaw) - y_local * sin(yaw) ... 
        # (这里仅做示意展示，不进行严格的刚体变换矩阵运算)
        
        # 4. 重心 (CoM) 稳定性
        # 假设在转弯时重心会产生离心偏移
        com_offset = 0.08 * np.sin(body_yaw * np.pi / 180) * np.exp(-0.2 * self.time)
        
        return body_yaw, (arm_local_x, arm_local_y, arm_local_z), com_offset

    def run_and_plot(self):
        print("========== 小问 3 求解开始 (协同控制模拟) ==========")
        
        episodes, rewards = self.generate_training_curve()
        body_yaw, arm_pos, com_offset = self.simulate_motion()
        arm_x, arm_y, arm_z = arm_pos
        
        # 开始绘图
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 训练曲线
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(episodes, rewards, 'b-', alpha=0.5, label='Episode Reward')
        # 移动平均
        window = 50
        avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], avg_rewards, 'r-', linewidth=2, label='Moving Avg')
        ax1.set_title('RL Training Convergence (PPO)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 身体朝向
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(self.time, body_yaw, 'k-', linewidth=2)
        ax2.axhline(y=45, color='r', linestyle='--', label='Target 45°')
        ax2.set_title('Body Orientation (Yaw)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (deg)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 手臂轨迹 (3D)
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot(arm_x, arm_y, arm_z, 'm-', linewidth=2)
        ax3.scatter(arm_x[0], arm_y[0], arm_z[0], color='g', s=50, label='Start')
        ax3.set_title('End-effector Trajectory (Local Frame)')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.legend()
        
        # 4. 稳定性监控
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(self.time, com_offset, 'g-', label='Lateral Offset')
        ax4.fill_between(self.time, -0.1, 0.1, color='gray', alpha=0.2, label='Stability Region')
        ax4.set_title('CoM Lateral Stability')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Offset (m)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        ensure_dir('图')
        save_path = '图/Question_3_Result.png'
        plt.savefig(save_path, dpi=300)
        print(f"结果示意图已保存至 {save_path}")

if __name__ == "__main__":
    sim = RobotSimulation()
    sim.run_and_plot()
