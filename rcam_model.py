import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ==================== PARAMETERS ====================
@dataclass
class Params:
    """Aircraft parameters container"""
    mass: float = 120000  # kg

    # Aerodynamic parameters
    cbar: float = 6.6
    l: float = 6.6  # mean aerodynamic chord
    lt: float = 24.6  # distance from CoG to AC of tail
    S: float = 260  # wing planform area
    St: float = 64  # tail planform area

    # Center of Gravity location
    xcg: float = 0.23 * 6.6
    ycg: float = 0.0
    zcg: float = 0.10 * 6.6

    # Aerodynamic Center location
    xac: float = 0.12 * 6.6
    yac: float = 0.0
    zac: float = 0.0

    # Engine attachment points
    xapt1: float = 0.0
    yapt1: float = -7.94
    zapt1: float = -1.9

    xapt2: float = 0.0
    yapt2: float = 7.94
    zapt2: float = -1.9

    # Environmental parameters
    rho: float = 1.225  # air density
    g: float = 9.81  # gravitational acceleration


params = Params()

# ==================== INITIALIZATION ====================
x_init = np.array([85, 0, 0, 0, 0, 0, 0, 0.1, 0])  # 9x1 state vector
u_init = np.array([0, -0.1, 0, 0.08, 0.08])  # 5x1 control vector

# Aileron: -0.1502° Elevator: -12.0574° Rudder: -0.3693° Throttle 1: 0.091787 Throttle 2: 0.095224
# u_init = np.array([-0.00262148, -0.210411, -0.006446, 0.091787, 0.095224])  # 5x1 control vector


# ==================== RCAM DYNAMICS MODEL ====================
def rcam(t: float, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Recursive aircraft model (RCAM) dynamics function.

    Args:
        t: time (not used, included for ODE solver compatibility)
        X: State vector [u, v, w, p, q, r, phi, theta, psi]
        U: Control input [aileron, elevator, rudder, throttle1, throttle2]

    Returns:
        XDot: State derivatives
    """

    # ==================== UNPACK STATE AND CONTROL ====================
    # Control inputs
    u1 = U[0]  # aileron
    u2 = U[1]  # elevator
    u3 = U[2]  # rudder
    u4 = U[3]  # throttle 1
    u5 = U[4]  # throttle 2

    # State variables
    x1 = X[0]  # u (forward velocity)
    x2 = X[1]  # v (lateral velocity)
    x3 = X[2]  # w (vertical velocity)
    x4 = X[3]  # p (roll rate)
    x5 = X[4]  # q (pitch rate)
    x6 = X[5]  # r (yaw rate)
    x7 = X[6]  # phi (roll angle)
    x8 = X[7]  # theta (pitch angle)
    x9 = X[8]  # psi (yaw angle)

    # ==================== AERODYNAMIC CONSTANTS ====================
    depsda = 0.25  # change in downwash wrt alpha
    alpha_L0 = -11.5 * np.pi / 180  # zero lift angle of attack
    n = 5.5  # linear region slope of lift curve
    a3 = -768.5  # coeff of alpha^3
    a2 = 609.2  # coeff of alpha^2
    a1 = -155.2  # coeff of alpha^1
    a0 = 15.212
    alpha_switch = 14.5 * (np.pi / 180)

    # ==================== CONTROL SATURATION ====================
    u1 = np.clip(u1, -25 * np.pi / 180, 10 * np.pi / 180)
    u2 = np.clip(u2, -25 * np.pi / 180, 10 * np.pi / 180)
    u3 = np.clip(u3, -30 * np.pi / 180, 30 * np.pi / 180)
    u4 = np.clip(u4, 0.5 * np.pi / 180, 10 * np.pi / 180)
    u5 = np.clip(u5, 0.5 * np.pi / 180, 10 * np.pi / 180)

    # ==================== INTERMEDIATE VARIABLES ====================
    # Airspeed calculation (accounting for wind)
    Va = np.sqrt(x1**2 + x2**2 + x3**2)

    # Angle of attack and sideslip
    alpha = np.arctan2(x3, x1)
    beta = np.arcsin(np.clip(x2 / Va, -1, 1))  # clip to avoid numerical issues

    # Dynamic pressure
    Q = 0.5 * params.rho * Va**2

    # Velocity and angular velocity vectors in body frame
    V_b = np.array([x1, x2, x3])
    wbe_b = np.array([x4, x5, x6])

    # ==================== AERODYNAMIC FORCE COEFFICIENTS ====================
    # Lift coefficient (wing and body)
    if alpha <= alpha_switch:
        CL_wb = n * (alpha - alpha_L0)
    else:
        CL_wb = a3 * alpha**3 + a2 * alpha**2 + a1 * alpha + a0

    # Tail aerodynamics
    epsilon = depsda * (alpha - alpha_L0)
    alpha_t = alpha - epsilon + u2 + 1.3 * x5 * params.lt / Va
    CL_t = 3.18 * (params.St / params.S) * alpha_t  # FIXED: Added missing * operator

    # Total lift coefficient
    CL = CL_wb + CL_t

    # Drag coefficient (neglecting tail)
    CD = 0.13 + 0.07 * (5.5 * alpha + 0.654)**2

    # Side force coefficient
    CY = -1.6 * beta + 0.24 * u3

    # ==================== DIMENSIONAL AERODYNAMIC FORCES ====================
    # Forces in stability axis frame
    FA_s = np.array([
        -CD * Q * params.S,
        CY * Q * params.S,
        -CL * Q * params.S
    ])

    # Rotation matrix from stability to body frame
    C_bs = np.array([
        [np.cos(alpha), 0, -np.sin(alpha)],
        [0, 1, 0],
        [np.sin(alpha), 0, np.cos(alpha)]
    ])

    # Forces in body frame
    FA_b = C_bs @ FA_s  # FIXED: Case consistency and proper matrix multiplication

    # ==================== AERODYNAMIC MOMENT COEFFICIENTS ====================
    eta11 = -1.4 * beta
    eta21 = -0.59 - (3.2 * (params.St / params.lt) / (params.S * params.l)) * (alpha - epsilon)
    eta31 = (1 - alpha * (180 / (15 * np.pi))) * beta
    eta = np.array([eta11, eta21, eta31])

    # Moment derivatives w.r.t. angular velocities
    dCMdx = (params.l / Va) * np.array([
        [-11, 0, 5],
        [0, -4.03 * (params.St * params.lt) / (params.S * params.l**2), 0],
        [1.7, 0, -11.5]
    ])

    # Moment derivatives w.r.t. control inputs
    dCMdu = np.array([
        [-0.6, 0, 0.22],
        [0, -3.1 * (params.St * params.lt) / (params.S * params.l), 0],
        [0, 0, -0.63]
    ])

    # Moment coefficients about aerodynamic center in body frame
    CMac_b = eta + dCMdx @ wbe_b + dCMdu @ np.array([u1, u2, u3])

    # ==================== AERODYNAMIC MOMENT ====================
    MAac_b = CMac_b * Q * params.S * params.l

    # ==================== TRANSFER MOMENT TO CG ====================
    rcg_b = np.array([params.xcg, params.ycg, params.zcg])
    rac_b = np.array([params.xac, params.yac, params.zac])

    MAcg_b = MAac_b + np.cross(FA_b, rcg_b - rac_b)

    # ==================== ENGINE FORCES ====================
    Tmax = params.mass * params.g
    F1 = u4 * Tmax
    F2 = u5 * Tmax

    # Engine thrust (assumed forward direction only)
    FE1_b = np.array([F1, 0, 0])
    FE2_b = np.array([F2, 0, 0])

    FE_b = FE1_b + FE2_b

    # ==================== ENGINE MOMENTS ====================
    # Moment arms from CG to engine attachment points
    mew1 = np.array([
        params.xcg - params.xapt1,
        params.yapt1 - params.ycg,
        params.zcg - params.zapt1
    ])

    mew2 = np.array([
        params.xcg - params.xapt2,  # FIXED: Was params.xch
        params.yapt2 - params.ycg,
        params.zcg - params.zapt2
    ])

    # Engine moments about CG
    MEcg1_b = np.cross(mew1, FE1_b)
    MEcg2_b = np.cross(mew2, FE2_b)

    MEcg_b = MEcg1_b + MEcg2_b

    # ==================== GRAVITY FORCES ====================
    # Gravity vector in body frame (causes no moment about CoG)
    g_b = np.array([
        -params.g * np.sin(x8),  # FIXED: Changed g to params.g
        params.g * np.cos(x8) * np.sin(x7),
        params.g * np.cos(x8) * np.cos(x7)
    ])

    Fg_b = params.mass * g_b

    # ==================== STATE DERIVATIVES ====================
    # Inertia matrix (constant)
    Ib = np.array([
        [40.07, 0, -2.0923],
        [0, 64, 0],
        [-2.0923, 0, 99.92]
    ]) # * params.mass

    # Inverse inertia matrix
    # invIb = (1 / params.mass) * np.array([
    #     [0.0249836, 0, 0.000523151],
    #     [0, 0.015625, 0],
    #     [0.000523151, 0, 0.0100191]
    # ])

    invIb = np.linalg.inv(Ib)

    # Total forces and linear accelerations
    F_b = Fg_b + FE_b + FA_b
    x1tox3dot = (1 / params.mass) * F_b - np.cross(wbe_b, V_b)

    # Total moments and angular accelerations
    Mcg_b = MAcg_b + MEcg_b
    x4tox6dot = invIb @ (Mcg_b - np.cross(wbe_b, Ib @ wbe_b))

    # Euler angle derivatives
    H_phi = np.array([
        [1, np.sin(x7) * np.tan(x8), np.cos(x7) * np.tan(x8)],
        [0, np.cos(x7), -np.sin(x7)],
        [0, np.sin(x7) / np.cos(x8), np.cos(x7) / np.cos(x8)]
    ])

    x7tox9dot = H_phi @ wbe_b

    # ==================== ASSEMBLE STATE DERIVATIVE VECTOR ====================
    XDot = np.concatenate([x1tox3dot, x4tox6dot, x7tox9dot])  # FIXED: Was x3tox6dor

    return XDot


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Test the dynamics function
    t = 0
    X = x_init.copy()
    U = u_init.copy()

    XDot = rcam(t, X, U)
    print("State derivatives at t=0:")
    print(XDot)
    print(f"\nState shape: {X.shape}")
    print(f"Control shape: {U.shape}")
    print(f"XDot shape: {XDot.shape}")
