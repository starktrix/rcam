import numpy as np
from find_equilibrium import find_equilibrium, find_equilibrium_multipoint
from plot_equilibrium import plot_equilibrium_results, plot_multipoint_analysis

IMG_PATH = 'equilibrium_images'
CMD = "equilibrium_cmds"

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":

    # Example 1: Single-point equilibrium at 85 m/s
    print("\n" + "="*70)
    print("EXAMPLE 1: Single-Point Equilibrium (85 m/s)")
    print("="*70)

    result_85 = find_equilibrium(target_airspeed=85.0, method='cost_only', verbose=True)

    print(f"\nOptimization Results:")
    print(f"  Success: {result_85['success']}")
    print(f"  Cost: {result_85['cost']:.6e}")
    print(f"  Max State Derivative: {result_85['max_derivative']:.6e}")
    print(f"\nOptimal Control Inputs:")
    print(f"  Aileron: {np.degrees(result_85['U_optimal'][0]):.4f}°")
    print(f"  Elevator: {np.degrees(result_85['U_optimal'][1]):.4f}°")
    print(f"  Rudder: {np.degrees(result_85['U_optimal'][2]):.4f}°")
    print(f"  Throttle 1: {result_85['U_optimal'][3]:.6f}")
    print(f"  Throttle 2: {result_85['U_optimal'][4]:.6f}")


    # Plot single-point result
    fig1 = plot_equilibrium_results(result_85)
    fig1.savefig(F'{IMG_PATH}/equilibrium_single_point.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to equilibrium_single_point.png")

    # Example 2: Multi-point analysis
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Point Equilibrium Analysis")
    print("="*70)

    airspeed_range = np.linspace(50, 120, 10)
    results_multipoint = find_equilibrium_multipoint(airspeed_range=airspeed_range, method='cost_only')

    # Plot multi-point analysis
    fig2 = plot_multipoint_analysis(results_multipoint)
    fig2.savefig(f'{IMG_PATH}/equilibrium_multipoint.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to equilibrium_multipoint.png")

    # Example 3: Find equilibrium at different airspeeds with better convergence
    print("\n" + "="*70)
    print("EXAMPLE 3: Additional Equilibrium Points")
    print("="*70)

    test_airspeeds = [70, 90, 110]
    for Va in test_airspeeds:
        result = find_equilibrium(target_airspeed=Va, method='cost_only', verbose=False)
        print(f"\nAirspeed {Va} m/s:")
        print(f"  Elevator: {np.degrees(result['U_optimal'][1]):.4f}°")
        print(f"  Throttle Avg: {np.mean(result['U_optimal'][3:5]):.6f}")
        print(f"  Cost: {result['cost']:.6e}")

    # Export equilibrium controls to CSV
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)

    export_data = np.array([[Va,
                             np.degrees(results_multipoint[Va]['U_optimal'][1]),
                             np.mean(results_multipoint[Va]['U_optimal'][3:5]),
                             results_multipoint[Va]['cost'],
                             results_multipoint[Va]['max_derivative']]
                            for Va in sorted(results_multipoint.keys())])

    header = 'airspeed_ms,elevator_deg,throttle_avg,cost,max_derivative'
    np.savetxt(f'{CMD}/equilibrium_commands.csv', export_data,
                delimiter=',', header=header, comments='', fmt='%.6f')
    print("Equilibrium commands saved to equilibrium_commands.csv")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)