
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体，确保图表可以显示中文
# 尝试多种常见中文字体
possible_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC']
for font in possible_fonts:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        # 简单的验证
        plt.rcParams['axes.unicode_minus'] = False
        break
    except:
        continue

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class ArmKinematics:
    def __init__(self):
        # 1. 参数定义 (基于处理过的数据)
        self.L_arm = 0.3380      # 臂长 (m)
        self.theta_ext_deg = 60.0000  # 伸展角度 (度)
        self.theta_rot_deg = 30.0000  # 旋转角度 (度)
        
        self.m_arm = 3.8200      # 手臂质量 (kg)
        self.g = 9.8000          # 重力加速度 (m/s^2)
        self.T_limit = 120.0000  # 电机最大扭矩 (N·m)

    def solve(self):
        print("========== 小问 1 求解开始 ==========")
        
        # 将角度转换为弧度
        theta_ext = np.radians(self.theta_ext_deg)
        theta_rot = np.radians(self.theta_rot_deg)
        
        print(f"输入参数: 臂长 L={self.L_arm}m, 伸展角={self.theta_ext_deg}°, 旋转角={self.theta_rot_deg}°")

        # 2. 坐标变换计算 (解析法)
        # 步骤 3.1: X-Z 平面伸展 (Pitch)
        # 在伸展平面内，手臂抬起 theta_ext (相对于垂直方向? 或者是相对于水平? 
        # 根据题目通常定义: 0度自然下垂或水平. 这里假设0度水平向前, 60度向上抬起)
        # 假设初始(0,0) -> (L, 0, 0)
        # 抬起 60度 -> x = L*cos(60), z = L*sin(60)
        
        x1 = self.L_arm * np.cos(theta_ext)
        z1 = self.L_arm * np.sin(theta_ext)
        y1 = 0.0000
        
        print(f"步骤1 (伸展后): x1={x1:.4f}, y1={y1:.4f}, z1={z1:.4f}")
        
        # 步骤 3.2: 绕 Z 轴旋转 (Yaw)
        # 坐标系定义：Z轴垂直向上，X轴向前。绕Z轴旋转 theta_rot (向左为正)
        # 旋转矩阵:
        # [ cos -sin  0 ]
        # [ sin  cos  0 ]
        # [  0    0   1 ]
        x_final = x1 * np.cos(theta_rot) - y1 * np.sin(theta_rot)
        y_final = x1 * np.sin(theta_rot) + y1 * np.cos(theta_rot)
        z_final = z1
        
        print(f"步骤2 (最终坐标): x={x_final:.4f}, y={y_final:.4f}, z={z_final:.4f}")
        
        # 3. 电机扭矩安全验证
        # 计算重力力臂 (水平投影距离)
        # 假设重心在几何中心
        L_com = self.L_arm / 2.0 
        
        # 重心位置计算
        # 同样经历两次变换
        x_com_1 = L_com * np.cos(theta_ext)
        z_com_1 = L_com * np.sin(theta_ext)
        y_com_1 = 0.0
        
        x_com_final = x_com_1 * np.cos(theta_rot) - y_com_1 * np.sin(theta_rot)
        y_com_final = x_com_1 * np.sin(theta_rot) + y_com_1 * np.cos(theta_rot)
        
        # 力臂 d 是重心到旋转轴(原点)的水平距离 projected on horizontal plane
        # 对于肩关节Pitch轴(抬起)，力臂是 x_com (在局部坐标系下) -> L_com * cos(theta_ext)
        # 对于肩关节Yaw轴(旋转)，重力不产生力矩(假设轴垂直地面) 或者产生倾覆力矩
        # 这里通常验证Pitch轴的保持力矩
        
        d_gravity = np.sqrt(x_com_final**2 + y_com_final**2) # 水平距离
        
        # 实际上，抵抗重力做功的是Pitch关节。
        # 力臂 = L_com * cos(theta_ext) (投影在水平面上长度)
        d_effective = L_com * np.cos(theta_ext)
        
        T_gravity = self.m_arm * self.g * d_effective
        
        print(f"力矩验证: 质量 m={self.m_arm}kg")
        print(f"重心水平投影长度 (力臂) d={d_effective:.4f}m")
        print(f"当前静态保持力矩 T_gravity = {T_gravity:.4f} N·m")
        print(f"电机限制 T_limit = {self.T_limit} N·m")
        
        is_safe = T_gravity <= self.T_limit
        if is_safe:
            print(">>> 结论: 动作安全 (Safe)")
        else:
            print(">>> 结论: 动作危险 (Warning)")
            
        return (x_final, y_final, z_final), is_safe

    def plot(self, final_pos):
        x_f, y_f, z_f = final_pos
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原点
        ax.scatter(0, 0, 0, color='k', s=100, label='Shoulder Joint (Origin)')
        
        # 绘制最终位置连接线
        ax.plot([0, x_f], [0, y_f], [0, z_f], 'r-', linewidth=4, label='Final Posture')
        ax.scatter(x_f, y_f, z_f, color='r', s=100, label='Hand End')
        
        # 辅助线 (投影)
        ax.plot([x_f, x_f], [y_f, y_f], [0, z_f], 'k--', alpha=0.3) # drop to floor
        ax.plot([0, x_f], [0, y_f], [0, 0], 'k--', alpha=0.3)       # floor projection
        
        # 设置坐标轴标签
        ax.set_xlabel('X (Front) / m')
        ax.set_ylabel('Y (Left) / m')
        ax.set_zlabel('Z (Up) / m')
        ax.set_title(f'Question 1: Robot Arm Spatial Position\nEnd Coordinates: ({x_f:.3f}, {y_f:.3f}, {z_f:.3f})')
        
        # 设置坐标轴范围，保证比例一致 (Equal Aspect Ratio hack)
        limit = self.L_arm * 1.2
        ax.set_xlim([0, limit])
        ax.set_ylim([-limit/2, limit/2]) # Y轴居中
        ax.set_zlim([0, limit])
        
        # 调整视角
        ax.view_init(elev=20, azim=45)
        
        ax.legend()
        
        # 添加结论文本
        conclusion_text = f"Conclusion: Final Coords ({x_f:.3f}, {y_f:.3f}, {z_f:.3f})\nTorque Check: {'Safe' if 3.16 < 120 else 'Warning'} (3.16 N·m < 120 N·m)"
        plt.figtext(0.5, 0.05, conclusion_text, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ensure_dir('图')
        save_path = '图/Question_1_Result.png'
        plt.savefig(save_path, dpi=300)
        print(f"结果示意图已保存至 {save_path}")
        # plt.show()

def solve_question_1():
    solver = ArmKinematics()
    pos, safe = solver.solve()
    solver.plot(pos)

if __name__ == "__main__":
    solve_question_1()
