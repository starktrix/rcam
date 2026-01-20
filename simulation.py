import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from rcam_model import rcam, params, x_init, u_init

# @title
# ==================== CONTROL INPUT PROFILE ====================
def get_control_input(t: float) -> np.ndarray:
    """
    Define control inputs as a function of time.
    This is a simple example; modify as needed for your scenario.
    """
    # u = u_init.copy()
    u = np.array([-0.00262148, -0.210411, -0.006446, 0.091787, 0.095224]) # equilibrum controls
    # if t < 2 or t > 4:
    #     u[1] = 0  # aileron command
    #     u[2] = 0

    # # Example: Introduce a step in elevator at t=10s
    # if t > 10:
    #     u[1] = -0.05  # elevator command

    return u

# ==================== ODE WRAPPER ====================
def rcam_wrapper(X: np.ndarray, t: float) -> np.ndarray:
    """
    Wrapper for ODE solver that provides time-varying control inputs.
    """
    U = get_control_input(t)
    return rcam(t, X, U)



# ==================== SIMULATION ====================
def simulation(t_start=0.0, t_end=200.0, dt=0.01) -> np.ndarray:
    teval= np.arange(t_start, t_end, dt)
    num_steps = len(teval)
    print("Starting RCAM simulation...")
    print(f"Simulation duration: {t_start} to {t_end} seconds")
    print(f"Number of steps: {num_steps}")

    # Run simulation using scipy's odeint
    # X_trajectory = odeint(rcam_wrapper, x_init, time)

    sol = solve_ivp(
        lambda t, x: rcam(t, x, get_control_input(t)),
        [t_start, t_end],
        x_init,
        method="Radau",
        #max_step=dt,
        #t_eval=teval,
        rtol=1e-6,
        atol=1e-9
    )

    time = sol.t
    X_trajectory = sol.y.T

    print("Success:", sol.success)
    print("Message:", sol.message)
    print("Final time reached:", sol.t[-1])
    print("Expected final time:", t_end)


    print(f"Simulation complete!")
    print(f"Trajectory shape: {X_trajectory}")
    print(f"Trajectory shape: {X_trajectory.shape} {time.shape}")

    # ==================== EXTRACT STATE COMPONENTS ====================
    u_vel = X_trajectory[:, 0]  # forward velocity
    v_vel = X_trajectory[:, 1]  # lateral velocity
    w_vel = X_trajectory[:, 2]  # vertical velocity
    p_rate = X_trajectory[:, 3]  # roll rate
    q_rate = X_trajectory[:, 4]  # pitch rate
    r_rate = X_trajectory[:, 5]  # yaw rate
    phi = X_trajectory[:, 6]    # roll angle
    theta = X_trajectory[:, 7]  # pitch angle
    psi = X_trajectory[:, 8]    # yaw angle

    # Calculate derived quantities
    airspeed = np.sqrt(u_vel**2 + v_vel**2 + w_vel**2)
    phi_deg = np.degrees(phi)
    theta_deg = np.degrees(theta)
    psi_deg = np.degrees(psi)

    # ==================== PRINT KEY RESULTS ====================
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Initial State: {x_init}")
    print(f"Initial Control: {u_init}")
    print(f"\nFinal State: {X_trajectory[-1]}")
    print(f"\nAirspeed Range: {airspeed.min():.2f} to {airspeed.max():.2f} m/s")
    print(f"Roll Angle Range: {phi_deg.min():.2f}° to {phi_deg.max():.2f}°")
    print(f"Pitch Angle Range: {theta_deg.min():.2f}° to {theta_deg.max():.2f}°")
    print(f"Yaw Angle Range: {psi_deg.min():.2f}° to {psi_deg.max():.2f}°")

    return np.array([u_vel, v_vel, w_vel, p_rate, q_rate, r_rate, phi_deg, theta_deg, psi_deg, airspeed, time])