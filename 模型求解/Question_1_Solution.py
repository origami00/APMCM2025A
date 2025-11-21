
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体，确保图表可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def solve_question_1():
    """
    小问1求解：空间直角坐标系转换与电机扭矩验证
    """
    print("========== 小问 1 求解开始 ==========")
    
    # 1. 参数定义
    L_arm = 338.0000  # 臂长 (mm)
    theta_ext_deg = 60.0000  # 伸展角度 (度)
    theta_rot_deg = 30.0000  # 旋转角度 (度)
    
    # 将角度转换为弧度
    theta_ext = np.radians(theta_ext_deg)
    theta_rot = np.radians(theta_rot_deg)
    
    print(f"输入参数: 臂长 L={L_arm}mm, 伸展角={theta_ext_deg}°, 旋转角={theta_rot_deg}°")

    # 2. 坐标变换计算 (解析法)
    # 步骤 3.1: X-Z 平面伸展 (Pitch)
    # 在伸展平面内，手臂抬起 theta_ext
    x1 = L_arm * np.cos(theta_ext)
    z1 = L_arm * np.sin(theta_ext)
    y1 = 0.0000
    
    print(f"步骤1 (伸展后): x1={x1:.2f}, y1={y1:.2f}, z1={z1:.2f}")
    
    # 步骤 3.2: 绕 Z 轴旋转 (Yaw)
    # 坐标系定义：Z轴垂直向上，初始X轴向前。绕Z轴旋转 theta_rot (向左为正)
    # 旋转矩阵应用:
    # x = x1 * cos(rot) - y1 * sin(rot)
    # y = x1 * sin(rot) + y1 * cos(rot)
    # z = z1
    x_final = x1 * np.cos(theta_rot) - y1 * np.sin(theta_rot)
    y_final = x1 * np.sin(theta_rot) + y1 * np.cos(theta_rot)
    z_final = z1
    
    print(f"步骤2 (最终坐标): x={x_final:.2f}, y={y_final:.2f}, z={z_final:.2f}")
    
    # 3. 电机扭矩安全验证
    # 参数
    m_arm = 3.8200  # 手臂质量 (kg), 
    g = 9.8000      # 重力加速度 (m/s^2)
    T_limit = 120.0000 # 电机最大扭矩 (N·m)
    
    # 计算重力力臂 (水平投影距离)
    # 假设重心在几何中心
    L_com = L_arm / 2.0000 / 1000.0000 # 转换为米
    # 重心坐标 (假设随手臂一起运动)
    # 重心水平距离 d_horiz 是重心到 Z 轴的垂直距离
    # d_horiz = sqrt(x_com^2 + y_com^2)
    # 其实就是 x1_com (在伸展后但在旋转前的 x 投影)
    d_horiz = L_com * np.cos(theta_ext)
    
    # 计算力矩
    T_gravity = m_arm * g * d_horiz
    
    print(f"力矩验证: 质量 m={m_arm}kg, 力臂 d={d_horiz:.4f}m")
    print(f"当前静态力矩 T_gravity = {T_gravity:.4f} N·m")
    print(f"电机限制 T_limit = {T_limit} N·m")
    
    is_safe = T_gravity <= T_limit
    if is_safe:
        print(">>> 结论: 动作安全 (Safe)")
    else:
        print(">>> 结论: 动作危险 (Warning)")
        
    # 4. 结果可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原点
    ax.scatter(0, 0, 0, color='k', s=100, label='肩关节 (原点)')
    
    # 绘制初始位置 (假设手臂自然下垂或水平向前，这里假设水平向前作为0位)
    # ax.plot([0, L_arm], [0, 0], [0, 0], 'k--', alpha=0.3, label='初始位置')
    
    # 绘制伸展后中间位置 (仅示意)
    # ax.plot([0, x1], [0, y1], [0, z1], 'b--', alpha=0.5, label='步骤1: 仅伸展')
    
    # 绘制最终位置
    ax.plot([0, x_final], [0, y_final], [0, z_final], 'r-', linewidth=3, label='最终姿态')
    ax.scatter(x_final, y_final, z_final, color='r', s=100, label='手部末端')
    
    # 辅助线
    ax.plot([x_final, x_final], [y_final, y_final], [0, z_final], 'k:', alpha=0.5)
    ax.plot([0, x_final], [0, y_final], [0, 0], 'k:', alpha=0.5)
    
    # 设置坐标轴标签
    ax.set_xlabel('X (前) / mm')
    ax.set_ylabel('Y (左) / mm')
    ax.set_zlabel('Z (上) / mm')
    ax.set_title(f'小问1: 机器人手臂空间位置示意\n末端坐标: ({x_final:.1f}, {y_final:.1f}, {z_final:.1f})')
    
    # 设置坐标轴范围，保证比例一致
    limit = L_arm * 1.2
    ax.set_xlim([0, limit])
    ax.set_ylim([-limit/2, limit])
    ax.set_zlim([0, limit])
    
    ax.legend()
    
    # 保存图片
    plt.savefig('模型求解/Question_1_Result.png')
    print("结果示意图已保存至 Question_1_Result.png")
    # plt.show() # 交互式环境可取消注释

if __name__ == "__main__":
    solve_question_1()

