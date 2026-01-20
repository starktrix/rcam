"""
RCAM Straight-Level Flight Equilibrium Finder
Finds control inputs (U) that achieve straight-level flight conditions
Uses scipy.optimize.minimize with trust-constr method
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from rcam_model import rcam, params, x_init, u_init
from cost_fn import equilibrium_cost, equilibrium_cost_with_state_error

# ==================== CONSTRAINT FUNCTIONS ====================

def state_derivative_constraint(U_opt: np.ndarray, target_airspeed: float = 85.0) -> np.ndarray:
    """
    Constraint: State derivatives should be zero for equilibrium.
    This is used as a nonlinear constraint in the optimization.

    Args:
        U_opt: Control inputs
        target_airspeed: Desired airspeed

    Returns:
        Array of state derivatives (should be ≈ 0)
    """
    X_eq = np.array([target_airspeed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X_dot = rcam(0, X_eq, U_opt)
    return X_dot


# ==================== OPTIMIZATION SETUP ====================

def find_equilibrium(target_airspeed: float = 85.0,
                     method: str = 'cost_only',
                     verbose: bool = True) -> dict:
    """
    Find straight-level flight equilibrium conditions.

    Args:
        target_airspeed: Desired airspeed (m/s)
        method: 'cost_only' or 'with_constraints'
        verbose: Print optimization details

    Returns:
        Dictionary with optimization results and equilibrium control/state
    """

    print(f"\n{'='*70}")
    print(f"RCAM EQUILIBRIUM FINDER - Straight-Level Flight")
    print(f"{'='*70}")
    print(f"Target Airspeed: {target_airspeed} m/s")
    print(f"Method: {method}")

    # Initial guess: use provided initial controls
    U0 = u_init.copy()

    # Bounds for control inputs (in radians and normalized thrust)
    bounds = [
        (-25*np.pi/180, 10*np.pi/180),   # u1: aileron
        (-25*np.pi/180, 10*np.pi/180),   # u2: elevator
        (-30*np.pi/180, 30*np.pi/180),   # u3: rudder
        (0.5*np.pi/180, 10*np.pi/180),   # u4: throttle 1
        (0.5*np.pi/180, 10*np.pi/180),   # u5: throttle 2
    ]

    # ==================== METHOD 1: Cost-only Minimization ====================
    if method == 'cost_only':
        print("\nUsing cost function minimization (derivatives ≈ 0)...")

        options = {
            'maxiter': 1000,
            'verbose': 1 if verbose else 0,
            'gtol': 1e-6,
        }

        result = minimize(
            equilibrium_cost,
            U0,
            args=(target_airspeed,),
            method='trust-constr',
            bounds=bounds,
            options=options
        )

    # ==================== METHOD 2: With Constraints ====================
    elif method == 'with_constraints':
        print("\nUsing constrained optimization (X_dot = 0 as constraint)...")

        # Nonlinear constraint: state derivatives should be zero
        constraint = NonlinearConstraint(
            lambda U: state_derivative_constraint(U, target_airspeed),
            np.zeros(9),  # lower bound
            np.zeros(9)   # upper bound
        )

        options = {
            'maxiter': 1000,
            'verbose': 1 if verbose else 0,
        }

        result = minimize(
            equilibrium_cost,
            U0,
            args=(target_airspeed,),
            method='trust-constr',
            bounds=bounds,
            constraints=constraint,
            options=options
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # ==================== EXTRACT AND VERIFY RESULTS ====================

    U_opt = result.x

    # Calculate equilibrium state and derivatives
    X_eq = np.array([target_airspeed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X_dot_eq = rcam(0, X_eq, U_opt)

    # Calculate actual airspeed with wind
    actual_airspeed = np.sqrt(
        X_eq[0]**2 +
        X_eq[1]**2 +
        X_eq[2]**2
    )

    # Package results
    results = {
        'success': result.success,
        'message': result.message,
        'nit': result.nit,
        'nfev': result.nfev,
        'cost': result.fun,
        'U_optimal': U_opt,
        'X_equilibrium': X_eq,
        'X_dot_equilibrium': X_dot_eq,
        'max_derivative': np.max(np.abs(X_dot_eq)),
        'target_airspeed': target_airspeed,
        'actual_airspeed': actual_airspeed,
    }

    return results


def find_equilibrium_multipoint(airspeed_range: np.ndarray = None,
                                method: str = 'cost_only') -> dict:
    """
    Find equilibrium conditions across a range of airspeeds.

    Args:
        airspeed_range: Array of target airspeeds to optimize
        method: Optimization method

    Returns:
        Dictionary with results for each airspeed
    """
    if airspeed_range is None:
        airspeed_range = np.linspace(50, 120, 10)

    results_dict = {}

    print(f"\n{'='*70}")
    print(f"MULTI-POINT EQUILIBRIUM ANALYSIS")
    print(f"{'='*70}")
    print(f"Analyzing {len(airspeed_range)} airspeeds from {airspeed_range[0]:.1f} to {airspeed_range[-1]:.1f} m/s\n")

    for i, Va in enumerate(airspeed_range):
        print(f"[{i+1}/{len(airspeed_range)}] Va = {Va:.1f} m/s...", end='', flush=True)

        result = find_equilibrium(target_airspeed=Va, method=method, verbose=False)
        results_dict[Va] = result

        status = "✓" if result['success'] else "✗"
        print(f" {status} (cost: {result['cost']:.6f}, max_deriv: {result['max_derivative']:.6e})")

    return results_dict

