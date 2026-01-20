import numpy as np
import matplotlib.pyplot as plt
from simulation import simulation

SAVE_DIR = "sim_plot"

# ==================== SIMULATION CONFIGURATION ====================
t_start = 0.0
t_end = 200.0
dt = 0.01

sim_result = simulation(t_start, t_end, dt)
u_vel = sim_result[0]
v_vel = sim_result[1]
w_vel = sim_result[2]
p_rate = sim_result[3]
q_rate = sim_result[4]
r_rate = sim_result[5]
phi_deg = sim_result[6]
theta_deg = sim_result[7]
psi_deg = sim_result[8]
airspeed = sim_result[9]
time = sim_result[10]
# ==================== PLOTTING ====================
fig = plt.figure(figsize=(16, 12))

# Velocities
ax1 = plt.subplot(3, 3, 1)
ax1.plot(time, u_vel, label='u (forward)', linewidth=1.5)
ax1.plot(time, v_vel, label='v (lateral)', linewidth=1.5)
ax1.plot(time, w_vel, label='w (vertical)', linewidth=1.5)
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Body-Frame Velocities')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Airspeed
ax2 = plt.subplot(3, 3, 2)
ax2.plot(time, airspeed, 'k-', linewidth=1.5)
ax2.set_ylabel('Airspeed (m/s)')
ax2.set_title('True Airspeed')
ax2.grid(True, alpha=0.3)

# Angular rates
ax3 = plt.subplot(3, 3, 3)
ax3.plot(time, np.degrees(p_rate), label='p (roll rate)', linewidth=1.5)
ax3.plot(time, np.degrees(q_rate), label='q (pitch rate)', linewidth=1.5)
ax3.plot(time, np.degrees(r_rate), label='r (yaw rate)', linewidth=1.5)
ax3.set_ylabel('Angular Rate (deg/s)')
ax3.set_title('Body-Frame Angular Rates')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Roll angle
ax4 = plt.subplot(3, 3, 4)
ax4.plot(time, phi_deg, 'b-', linewidth=1.5)
ax4.set_ylabel('Angle (degrees)')
ax4.set_title('Roll Angle (φ)')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Pitch angle
ax5 = plt.subplot(3, 3, 5)
ax5.plot(time, theta_deg, 'g-', linewidth=1.5)
ax5.set_ylabel('Angle (degrees)')
ax5.set_title('Pitch Angle (θ)')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Yaw angle
ax6 = plt.subplot(3, 3, 6)
ax6.plot(time, psi_deg, 'r-', linewidth=1.5)
ax6.set_ylabel('Angle (degrees)')
ax6.set_title('Yaw Angle (ψ)')
ax6.grid(True, alpha=0.3)

# Phase portrait: u vs w
ax7 = plt.subplot(3, 3, 7)
ax7.plot(u_vel, w_vel, 'b-', linewidth=1.5, alpha=0.7)
ax7.scatter([u_vel[0]], [w_vel[0]], color='g', s=100, label='Start', zorder=5)
ax7.scatter([u_vel[-1]], [w_vel[-1]], color='r', s=100, label='End', zorder=5)
ax7.set_xlabel('u (m/s)')
ax7.set_ylabel('w (m/s)')
ax7.set_title('Velocity Phase Portrait (u-w)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Angle of attack approximation
alpha_approx = np.arctan2(w_vel, u_vel)
ax8 = plt.subplot(3, 3, 8)
ax8.plot(time, np.degrees(alpha_approx), 'purple', linewidth=1.5)
ax8.set_ylabel('Angle (degrees)')
ax8.set_title('Approximate Angle of Attack')
ax8.grid(True, alpha=0.3)

# Summary statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
stats_text = f"""
Simulation Summary Statistics:

Initial Airspeed: {airspeed[0]:.2f} m/s
Final Airspeed: {airspeed[-1]:.2f} m/s
Min Airspeed: {airspeed.min():.2f} m/s
Max Airspeed: {airspeed.max():.2f} m/s

Final φ: {phi_deg[-1]:.2f}°
Final θ: {theta_deg[-1]:.2f}°
Final ψ: {psi_deg[-1]:.2f}°

Simulation Time: {t_end - t_start:.1f} s
Number of Points: {len(time)}
"""
ax9.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/rcam_simulation.png', dpi=150, bbox_inches='tight')
print("Plot saved to rcam_simulation.png")
plt.show()

# ==================== DATA EXPORT ====================
# Save trajectory to CSV for further analysis
data = np.column_stack([
    time, u_vel, v_vel, w_vel, p_rate, q_rate, r_rate,
    phi_deg, theta_deg, psi_deg, airspeed, np.degrees(alpha_approx)
])

header = 'time,u,v,w,p,q,r,phi_deg,theta_deg,psi_deg,airspeed,alpha_deg'
np.savetxt(f'{SAVE_DIR}/rcam_trajectory.csv', data, delimiter=',', header=header, comments='')
print("Trajectory saved to rcam_trajectory.csv")