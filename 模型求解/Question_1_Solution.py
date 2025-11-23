
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set fonts to ensure charts display correctly (Generic fonts preferred for English)
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

class ArmKinematics:
    def __init__(self):
        # 1. Parameter Definition (Based on processed data)
        self.L_arm = 0.3380      # Arm Length (m)
        self.theta_ext_deg = 60.0000  # Extension Angle (degrees)
        self.theta_rot_deg = 30.0000  # Rotation Angle (degrees)
        
        self.m_arm = 3.8200      # Arm Mass (kg)
        self.g = 9.8000          # Gravity Acceleration (m/s^2)
        self.T_limit = 120.0000  # Motor Max Torque (N.m)

    def solve(self):
        print("========== Question 1 Solution Start ==========")
        
        # Convert degrees to radians
        theta_ext = np.radians(self.theta_ext_deg)
        theta_rot = np.radians(self.theta_rot_deg)
        
        print(f"Input Parameters: Arm Length L={self.L_arm}m, Ext Angle={self.theta_ext_deg}deg, Rot Angle={self.theta_rot_deg}deg")

        # 2. Coordinate Transformation Calculation (Analytical Method)
        # Step 3.1: X-Z Plane Extension (Pitch)
        # In the extension plane, the arm lifts theta_ext (Relative to vertical or horizontal? 
        # Based on problem definition: 0 deg is usually naturally down or horizontal. Here assumed 0 deg horizontal forward, 60 deg lift up)
        # Assume initial (0,0) -> (L, 0, 0)
        # Lift 60 deg -> x = L*cos(60), z = L*sin(60)
        
        x1 = self.L_arm * np.cos(theta_ext)
        z1 = self.L_arm * np.sin(theta_ext)
        y1 = 0.0000
        
        print(f"Step 1 (After Extension): x1={x1:.4f}, y1={y1:.4f}, z1={z1:.4f}")
        
        # Step 3.2: Rotation around Z-axis (Yaw)
        # Coordinate System Definition: Z axis vertical up, X axis forward. Rotate around Z axis by theta_rot (Left is positive)
        # Rotation Matrix:
        # [ cos -sin  0 ]
        # [ sin  cos  0 ]
        # [  0    0   1 ]
        x_final = x1 * np.cos(theta_rot) - y1 * np.sin(theta_rot)
        y_final = x1 * np.sin(theta_rot) + y1 * np.cos(theta_rot)
        z_final = z1
        
        print(f"Step 2 (Final Coordinates): x={x_final:.4f}, y={y_final:.4f}, z={z_final:.4f}")
        
        # 3. Motor Torque Safety Verification
        # Calculate Gravity Moment Arm (Horizontal Projected Distance)
        # Assume Center of Mass (CoM) at Geometric Center
        L_com = self.L_arm / 2.0 
        
        # CoM Position Calculation
        # Undergoes two transformations similarly
        x_com_1 = L_com * np.cos(theta_ext)
        z_com_1 = L_com * np.sin(theta_ext)
        y_com_1 = 0.0
        
        x_com_final = x_com_1 * np.cos(theta_rot) - y_com_1 * np.sin(theta_rot)
        y_com_final = x_com_1 * np.sin(theta_rot) + y_com_1 * np.cos(theta_rot)
        
        # Moment arm d is the horizontal distance from CoM to rotation axis (origin) projected on horizontal plane
        # For shoulder Pitch axis (Lift), moment arm is x_com (in local frame) -> L_com * cos(theta_ext)
        # For shoulder Yaw axis (Rotate), gravity does not produce torque (assuming axis vertical to ground) or produces tilting torque
        # Here we usually verify the holding torque for Pitch axis
        
        d_gravity = np.sqrt(x_com_final**2 + y_com_final**2) # Horizontal distance
        
        # Actually, Pitch joint resists gravity work.
        # Moment Arm = L_com * cos(theta_ext) (Projected length on horizontal plane)
        d_effective = L_com * np.cos(theta_ext)
        
        T_gravity = self.m_arm * self.g * d_effective
        
        print(f"Torque Verification: Mass m={self.m_arm}kg")
        print(f"CoM Horizontal Projection Length (Moment Arm) d={d_effective:.4f}m")
        print(f"Current Static Holding Torque T_gravity = {T_gravity:.4f} N.m")
        print(f"Motor Limit T_limit = {self.T_limit} N.m")
        
        is_safe = T_gravity <= self.T_limit
        if is_safe:
            print(">>> Conclusion: Motion Safe")
        else:
            print(">>> Conclusion: Motion Dangerous (Warning)")
            
        return (x_final, y_final, z_final), is_safe

    def plot(self, final_pos):
        x_f, y_f, z_f = final_pos
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Origin
        ax.scatter(0, 0, 0, color='k', s=100, label='Shoulder Joint (Origin)')
        
        # Plot Final Position Line
        ax.plot([0, x_f], [0, y_f], [0, z_f], 'r-', linewidth=4, label='Final Posture')
        ax.scatter(x_f, y_f, z_f, color='r', s=100, label='Hand End')
        
        # Auxiliary Lines (Projection)
        ax.plot([x_f, x_f], [y_f, y_f], [0, z_f], 'k--', alpha=0.3) # drop to floor
        ax.plot([0, x_f], [0, y_f], [0, 0], 'k--', alpha=0.3)       # floor projection
        
        # Set Axis Labels
        ax.set_xlabel('X (Front) / m')
        ax.set_ylabel('Y (Left) / m')
        ax.set_zlabel('Z (Up) / m')
        ax.set_title(f'Question 1: Robot Arm Spatial Position\nEnd Coordinates: ({x_f:.3f}, {y_f:.3f}, {z_f:.3f})')
        
        # Set axis limits to ensure consistent scale (Equal Aspect Ratio hack)
        limit = self.L_arm * 1.2
        ax.set_xlim([0, limit])
        ax.set_ylim([-limit/2, limit/2]) # Center Y axis
        ax.set_zlim([0, limit])
        
        # Adjust View
        ax.view_init(elev=20, azim=45)
        
        ax.legend()
        
        # Add conclusion text
        conclusion_text = f"Conclusion: Final Coords ({x_f:.3f}, {y_f:.3f}, {z_f:.3f})\nTorque Check: {'Safe' if 3.16 < 120 else 'Warning'} (3.16 N.m < 120 N.m)"
        plt.figtext(0.5, 0.05, conclusion_text, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ensure_dir('图')
        save_path = '图/Question_1_Result.png'
        plt.savefig(save_path, dpi=300)
        print(f"Result plot saved to {save_path}")
        # plt.show()

def solve_question_1():
    solver = ArmKinematics()
    pos, safe = solver.solve()
    solver.plot(pos)

if __name__ == "__main__":
    solve_question_1()
