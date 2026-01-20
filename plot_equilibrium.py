import numpy as np
import matplotlib.pyplot as plt

# ==================== PLOTTING UTILITIES ====================

def plot_equilibrium_results(result: dict, figsize: tuple = (14, 10)):
    """
    Plot the equilibrium condition results.

    Args:
        result: Dictionary from find_equilibrium()
        figsize: Figure size
    """

    U_opt = result['U_optimal']
    X_dot_eq = result['X_equilibrium']
    cost = result['cost']

    fig = plt.figure(figsize=figsize)

    # Control inputs
    ax1 = plt.subplot(2, 3, 1)
    labels = ['Aileron', 'Elevator', 'Rudder', 'Throttle 1', 'Throttle 2']
    colors = ['b', 'g', 'r', 'orange', 'purple', 'brown', 'pink', 'gray']

    x_pos = np.arange(len(labels))
    heights = np.concatenate([np.degrees(U_opt[:3]),  [U_opt[3]*180/np.pi, U_opt[4]*180/np.pi]])
    ax1.bar(x_pos[:5], heights,
            color=colors[:5], alpha=0.7)
    ax1.set_xticks(x_pos[:5])
    ax1.set_xticklabels(labels[:5], rotation=45, ha='right')
    ax1.set_ylabel('Control Input (deg or normalized)')
    ax1.set_title('Optimal Control Inputs (First 5)')
    ax1.grid(True, alpha=0.3)


    # State derivatives
    ax3 = plt.subplot(2, 3, 2)
    state_labels = ['u', 'v', 'w', 'p', 'q', 'r', 'φ', 'θ', 'ψ']
    colors_state = ['b', 'g', 'r', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    ax3.bar(np.arange(len(state_labels)), X_dot_eq, color=colors_state, alpha=0.7)
    ax3.set_xticks(np.arange(len(state_labels)))
    ax3.set_xticklabels(state_labels)
    ax3.set_ylabel('State Derivative')
    ax3.set_title('State Derivatives at Equilibrium\n(Should be ≈ 0)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)

    # Optimization metrics
    ax4 = plt.subplot(2, 3, 3)
    ax4.axis('off')
    metrics_text = f"""
OPTIMIZATION RESULTS

Target Airspeed: {result['target_airspeed']:.2f} m/s
Actual Airspeed: {result['actual_airspeed']:.2f} m/s

Optimization Status: {'SUCCESS' if result['success'] else 'FAILED'}
Cost Function: {result['cost']:.6e}
Max Derivative: {result['max_derivative']:.6e}
Iterations: {result['nit']}
Function Evaluations: {result['nfev']}

Message: {result['message']}
"""
    ax4.text(0.1, 0.5, metrics_text, fontfamily='monospace', fontsize=9,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Control saturation check
    ax5 = plt.subplot(2, 3, 4)
    bounds_lower = [-25, -25, -30, 0.5, 0.5]
    bounds_upper = [10, 10, 30, 10, 10]

    U_plot = np.concatenate([
        np.degrees(U_opt[:3]),
        U_opt[3:5],
    ])
    bounds_lower_plot = np.array(bounds_lower[:3] + bounds_lower[3:])
    bounds_upper_plot = np.array(bounds_upper[:3] + bounds_upper[3:])

    x_pos = np.arange(len(U_plot))
    ax5.barh(x_pos, bounds_upper_plot - bounds_lower_plot, left=bounds_lower_plot,
             alpha=0.3, color='gray', label='Constraint Range')
    ax5.scatter(U_plot, x_pos, color='red', s=100, zorder=5, label='Optimal Value')
    ax5.set_yticks(x_pos)
    ax5.set_yticklabels(labels)
    ax5.set_xlabel('Control Value (deg or normalized)')
    ax5.set_title('Control Saturation Check')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_multipoint_analysis(results_dict: dict):
    """
    Plot equilibrium conditions across multiple airspeeds.

    Args:
        results_dict: Dictionary from find_equilibrium_multipoint()
    """

    airspeeds = sorted(results_dict.keys())

    # Extract data
    elevator_commands = []
    throttle_commands = []
    costs = []
    max_derivatives = []

    for Va in airspeeds:
        result = results_dict[Va]
        elevator_commands.append(np.degrees(result['U_optimal'][1]))
        throttle_avg = np.mean(np.degrees(result['U_optimal'][3:5]))
        throttle_commands.append(throttle_avg)
        costs.append(result['cost'])
        max_derivatives.append(result['max_derivative'])

    fig = plt.figure(figsize=(14, 10))

    # Elevator command
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(airspeeds, elevator_commands, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Airspeed (m/s)')
    ax1.set_ylabel('Elevator Command (deg)')
    ax1.set_title('Equilibrium Elevator Command vs Airspeed')
    ax1.grid(True, alpha=0.3)

    # Throttle command
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(airspeeds, throttle_commands, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Airspeed (m/s)')
    ax2.set_ylabel('Throttle Command (deg equivalent)')
    ax2.set_title('Equilibrium Throttle Command vs Airspeed')
    ax2.grid(True, alpha=0.3)

    # Cost function
    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogy(airspeeds, costs, 'r-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Airspeed (m/s)')
    ax3.set_ylabel('Cost Function')
    ax3.set_title('Optimization Cost vs Airspeed')
    ax3.grid(True, alpha=0.3)

    # Max derivative
    ax4 = plt.subplot(2, 2, 4)
    ax4.semilogy(airspeeds, max_derivatives, 'purple', marker='o', linewidth=2, markersize=6)
    ax4.set_xlabel('Airspeed (m/s)')
    ax4.set_ylabel('Max |X_dot|')
    ax4.set_title('Maximum State Derivative vs Airspeed')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
