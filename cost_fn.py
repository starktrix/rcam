import numpy as np
from rcam_model import rcam, params, x_init, u_init

# ==================== EQUILIBRIUM COST FUNCTION ====================

def equilibrium_cost(U_opt: np.ndarray, target_airspeed: float = 85.0) -> float:
    """
    Cost function for equilibrium: minimize state derivatives.

    For straight-level flight, we want:
    - Pitch rate (q) ≈ 0
    - Roll rate (p) ≈ 0
    - Yaw rate (r) ≈ 0
    - Pitch angle (theta) ≈ 0 (level flight)
    - Roll angle (phi) ≈ 0 (wings level)
    - Vertical velocity (w) ≈ 0

    Args:
        U_opt: Control inputs [aileron, elevator, rudder, throttle1, throttle2]
        target_airspeed: Desired airspeed in m/s

    Returns:
        Cost (float): Sum of squared state derivatives (penalizing motion)
    """

    # State for equilibrium: cruising at target airspeed, level, wings level
    X_eq = np.array([
        target_airspeed,  # u: forward velocity
        0.0,              # v: lateral velocity
        0.0,              # w: vertical velocity
        0.0,              # p: roll rate
        0.0,              # q: pitch rate
        0.0,              # r: yaw rate
        0.0,              # phi: roll angle
        0.0,              # theta: pitch angle
        0.0               # psi: yaw angle (doesn't matter for straight flight)
    ])

    # Calculate state derivatives at this condition
    t = 0
    X_dot = rcam(t, X_eq, U_opt)

    # Cost: weighted sum of squared derivatives
    # Prioritize zeroing angular rates and vertical velocity
    weights = np.array([
        10.0,   # u_dot: maintain airspeed
        50.0,   # v_dot: minimize lateral velocity change
        50.0,   # w_dot: minimize vertical velocity change
        100.0,  # p_dot: minimize roll rate change
        100.0,  # q_dot: minimize pitch rate change
        100.0,  # r_dot: minimize yaw rate change
        10.0,   # phi_dot: (should be zero naturally from p=0)
        10.0,   # theta_dot: (should be zero naturally from q=0)
        1.0     # psi_dot: (doesn't matter for straight flight)
    ])

    cost = np.sum(weights * X_dot**2)
    return cost


def equilibrium_cost_with_state_error(U_and_state: np.ndarray, target_airspeed: float = 85.0) -> float:
    """
    Alternative cost function that also penalizes deviations from desired state.
    Allows finding equilibrium at specific flight conditions.

    Args:
        U_and_state: [U (8D), X (9D)] concatenated vector
        target_airspeed: Desired airspeed

    Returns:
        Cost combining state error and derivatives
    """
    U_opt = U_and_state[:8]
    X_opt = U_and_state[8:17]  # Only first 9 states (not 12)

    # Desired state: straight-level flight at target airspeed
    X_desired = np.array([
        target_airspeed,  # u
        0.0,              # v
        0.0,              # w
        0.0,              # p
        0.0,              # q
        0.0,              # r
        0.0,              # phi
        0.0,              # theta
        0.0               # psi
    ])

    # State error cost
    state_weights = np.array([5.0, 50.0, 50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 1.0])
    state_error_cost = np.sum(state_weights * (X_opt - X_desired)**2)

    # Derivative cost
    X_dot = rcam(0, X_opt, U_opt)
    derivative_weights = np.array([10.0, 50.0, 50.0, 100.0, 100.0, 100.0, 10.0, 10.0, 1.0])
    derivative_cost = np.sum(derivative_weights * X_dot**2)

    total_cost = state_error_cost + derivative_cost
    return total_cost
